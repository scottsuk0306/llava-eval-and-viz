# Word and Image Vector Visualization with Llava

This code demonstrates how to extract and visualize word and image vectors using the Llava model. It provides functionality to generate word and image vectors for a predefined set of words, and then visualizes these vectors using t-SNE (t-Distributed Stochastic Neighbor Embedding).

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Pillow
- Requests
- Langdetect
- Scikit-learn
- Matplotlib
- Seaborn


## Usage

1. Prepare the necessary assets:
   - Place the image files corresponding to the words in the `assets` directory. The image files should be named `<word>.jpg` (e.g., `city.jpg`, `tree.jpg`, etc.).

2. Run the `generate_word_and_image_vectors()` function to generate word and image vectors:

```python
if __name__ == "__main__":
    generate_word_and_image_vectors()
```

This function will load the Llava model, process the images and words, and generate the corresponding vectors. The generated vectors will be saved in a pickle file named `word_and_image_vectors.pkl`.

3. Run the `visualize()` function to visualize the word and image vectors using t-SNE:

```python
if __name__ == "__main__":
    visualize()
```

This function will load the previously generated vectors from the pickle file, apply t-SNE dimensionality reduction, and visualize the vectors in a scatter plot. The resulting visualization will be saved as `tsne.png`.

## Customization

- To modify the set of words used for generating vectors, update the `words` list in the `generate_word_and_image_vectors()` function.
- To change the image paths, modify the `image_paths` list in the `generate_word_and_image_vectors()` function.
- The `CustomLlavaForConditionalGeneration` class extends the `LlavaForConditionalGeneration` class and overrides the `_merge_input_ids_with_image_features()` method to store the generated word and image vectors.
- The `is_english()` function is used to check if a word is in English using the Langdetect library.
- The `get_words_ids()` function returns dictionaries mapping words to their corresponding IDs and vice versa.

## Acknowledgments

- This code utilizes the Llava model and the Transformers library.
- The t-SNE visualization is implemented using Scikit-learn and Matplotlib.

## License

This project is licensed under the MIT Licence.