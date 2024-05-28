import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch
from langdetect import LangDetectException, detect
from PIL import Image
from sklearn.manifold import TSNE
from transformers import (AutoProcessor, AutoTokenizer, CLIPVisionConfig,
                          LlamaConfig, LlavaConfig,
                          LlavaForConditionalGeneration)


def is_english(word):
    try:
        return detect(word) == "en"
    except LangDetectException:
        return False


def get_words_ids():
    words_to_ids = {
        "city": 4272,
        "tree": 5447,
        "mouth": 13394,
        "heart": 5192,
        "fire": 3974,
        "castle": 20610,
        "people": 2305,
        "anonymous": 21560,
        "father": 4783,
        "office": 8034,
        "software": 7047,
        "football": 5733,
        "ear": 2326,
        "cat": 6635,
        "dog": 11203,
        "friend": 5121,
        "cow": 21282,
        "phone": 9008,
        "game": 3748,
        "person": 2022,
    }
    ids_to_words = {
        4272: "city",
        5447: "tree",
        13394: "mouth",
        5192: "heart",
        3974: "fire",
        20610: "castle",
        2305: "people",
        21560: "anonymous",
        4783: "father",
        8034: "office",
        7047: "software",
        5733: "football",
        2326: "ear",
        6635: "cat",
        11203: "dog",
        5121: "friend",
        21282: "cow",
        9008: "phone",
        3748: "game",
        2022: "person",
    }

    return words_to_ids, ids_to_words


word_and_image_vectors = {}


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, labels
    ):

        _, ids_to_words = get_words_ids()
        word = ids_to_words[int(input_ids[0][-1])]

        # print(image_features.shape)
        # print(inputs_embeds.shape)

        # image_vector = image_features.mean(dim=1)
        image_vector = image_features.max(dim=1)[0]
        # print(image_vector.shape)
        image_vector.reshape(1, 4096)

        word_vector = inputs_embeds[0][-1]
        # print(word_vector.shape)
        word_vector.reshape(1, 4096)

        assert image_vector.shape == torch.Size([1, 4096])
        assert word_vector.shape == torch.Size([4096])

        global word_and_image_vectors

        word_and_image_vectors[word] = {
            "image": image_vector,
            "word": word_vector,
        }

        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.pad_token_id)
        )
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_image_tokens.max() * (num_image_patches - 1)
        ) + sequence_length
        batch_indices, non_image_indices = torch.where(
            input_ids != self.config.image_token_index
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1)
            - 1
        )
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_embed_dim,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim),
                self.config.ignore_index,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_image_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_image_indices
            ]

        # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[
            :, None
        ].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = (
            image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


def generate_word_and_image_vectors():

    words = [
        "city",
        "tree",
        "mouth",
        "heart",
        "fire",
        "castle",
        "people",
        "anonymous",
        "father",
        "office",
        "software",
        "football",
        "ear",
        "cat",
        "dog",
        "friend",
        "cow",
        "phone",
        "game",
        "person",
    ]

    base_path = Path(__file__).parent
    image_paths = [base_path / f"assets/{word}.jpg" for word in words]

    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = CustomLlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf"
    )
    model.to("cuda")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    for idx, word in enumerate(words):
        prompt = f"<image> {word}"
        image = Image.open(image_paths[idx])
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs.to("cuda")
        generate_ids = model.generate(**inputs, max_new_tokens=15)
        print(
            processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        )

    global word_and_image_vectors

    with open("word_and_image_vectors.pkl", "wb") as f:
        pickle.dump(word_and_image_vectors, f)


def visualize():
    global word_and_image_vectors

    with open("word_and_image_vectors_mean.pkl", "rb") as f:
        word_and_image_vectors = pickle.load(f)

    words_to_ids, ids_to_words = get_words_ids()

    # Initialize arrays for data
    vectors = []
    labels = []
    base_words = []  # List to keep track of the base words without suffix

    # Collect vectors and their labels
    for word, data in word_and_image_vectors.items():
        vectors.append(
            data["word"].cpu().detach().numpy()
        )  # Assuming 'word' is a vector
        labels.append(word + " - Word")
        base_words.append(word)

        vectors.append(
            data["image"][0].cpu().detach().numpy()
        )  # Assuming first image vector
        labels.append(word + " - Image")
        base_words.append(word)

    # Convert list to numpy array
    vectors = np.array(vectors)

    # Create a color palette based on base words
    unique_words = list(set(base_words))
    palette = sns.color_palette("hsv", len(unique_words))
    color_map = dict(zip(unique_words, palette))

    # Apply t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=5, n_iter=3000)
    tsne_results = tsne.fit_transform(vectors)

    # Plotting
    plt.figure(figsize=(16, 10))
    for i, label in enumerate(labels):
        x, y = tsne_results[i, :]
        base_word = base_words[i]  # Get the base word for the label
        plt.scatter(
            x, y, color=color_map[base_word]
        )  # Use the base word to get the color
        plt.annotate(
            label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    plt.title("t-SNE visualization of Word and Image Vectors")
    plt.xlabel("t-SNE Axis 1")
    plt.ylabel("t-SNE Axis 2")
    plt.savefig("tsne.png")


def test_config():
    vision_config = CLIPVisionConfig()
    text_config = LlamaConfig()
    configuration = LlavaConfig(vision_config, text_config)
    model = LlavaForConditionalGeneration(configuration)
    configuration = model.config
    print(configuration)


if __name__ == "__main__":
    generate_word_and_image_vectors()
    visualize()
