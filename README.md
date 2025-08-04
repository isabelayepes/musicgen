# Music Generation with Transformers

A transformer-based model for generating short music snippets, originally implemented as a group project for the AC 209B Data Science 2 course.

## Sample Outputs
- Generated samples available in `audio/`

## Start
- You can use the `data/test` provided data subset or provided cached files in `model_weight/` for quicker generation, otherwise you can download more data from the data source [Lakh MIDI Dataset v0.1](https://colinraffel.com/projects/lmd/) (Raffel, 2016)
- Run the `local-colab-musicgen.ipynb`

## Features
- Python and PyTorch
- Colab and local environment support
- Data filtration for top 5 genres and 10 seconds or more music duration
- Automatic cache saving/loading
- Data visualization and charts even for cached data
- Custom transformer architecture class
- Genre conditioned tokenization and generation
- Evaluation of frechet distance, musical metrics, and KL divergence across genres

## Dependencies
- See `requirements.txt` for Python environment dependencies

## Note
Training takes significant time - use provided weights in `model_weights/` for quick testing.

## Data sources and background

The [Lakh MIDI Dataset v0.1](https://colinraffel.com/projects/lmd/) (Raffel, 2016) is a collection of 176,581 unique MIDI files of symbolic music data, 45,129 of which have been matched and aligned to entries in the Million Song Dataset. The MIDI files were scraped from publicly available sources on the internet and then de-duplicated according to their MD5 checksum. We will start with a subset of the LMD-matched, which is a curated subset of the larger LMD-full MIDI collection containing 45,129 files. The subset we are using for this analysis contains 4166 files. Each file has been carefully "matched" to an entry in the Million Song Dataset (MSD). This matching allows you to connect symbolic music data (MIDI) with rich metadata and audio previews available in the MSD.

The entire Lakh MIDI matched dataset **5.12GB** can be downloaded from the website. A selected subset `I/` of **170MB** has been zipped and uploaded to `data/test`. After downloading the full data, you would see it is split into alphabetical folders `A, B, ..., Z`. Inside of each of these folders is another group of alphabetical folders `A, B, ..., Z`, and there is one more nested layer after that again `A, B, ..., Z`. There is no structure to how MIDI files get mapped to one of these subdirectories, so, we chose one of them for our training data. In the pre-processing included in the `local-collab-music-gen.ipynb`, we ensure that all the corrupted, unreadable, or duplicates files are handled properly.

The LMD-matched subset provides structured metadata through its alignment with MSD, including fields like artist, title, and release year. However, this metadata is not directly relevant to our project. Instead, for metadata, we use [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps) dataset (Melechovsky, 2023) from HuggingFace, which offers more musically meaningful information about the MIDI files for music generation. MidiCaps is a textual `.json` file of **376MB** containing information such as instruments, genres, chords and tempo. The file `extract-json-hug-face.ipynb` includes code to extract the metadata. The metadata corresponds to the full LMD data and may not perfectly align with matched LMD data.

The missing data in this dataset appears to be Missing Completely at Random (MCAR). Based on the Hugging Face Dataset's description, there was no targeted filtering to exclude music based on specific criteria such as genre, instrumentation, or popularity. This indicates that missing files or metadata likely result from random factors like file availability or scraping limitations, rather than systematic exclusion. To minimize issues from missing data, we perform an **inner join** between the MIDI note statistics and the MidiCaps data, ensuring that we only retain files present in both sources. Since the missingness is assumed to be MCAR, this approach should not introduce bias into our results or analyses.

Data imbalance is a known issue, as different genres are not equally represented. MIDI files also vary in length, which introduces another form of imbalance. This can potentially be addressed through padding, truncation, or filtering of files before they are input into the music generation model. The number of instruments in a song is another potential source of imbalance but it was not addressed since the performance of the generation model seemed to effectively generate the sounds of multiple instruments. Our data filtration consisted of top 5 genres and 10 seconds or more music duration. In the future other forms of preprocessing could be considered.

Since our end goal was to build a music generation model, feature scaling was not directly applicable in the traditional sense, since we were working with symbolic music data rather than continuous numerical features. We did not train the model on the textual metadata. Models such as Transformers, LSTMs, and GANs typically operate on discrete event sequences (e.g., note on/off events, pitch values, velocity changes, time shifts), which are either tokenized into vocabulary-like inputs (e.g., MIDI-like or REMI format) or embedded via learned embeddings. Each symbolic event is treated as a categorical token, and embeddings learn the relationships between these tokens. As a result, standard feature scaling techniques such as Z-score normalization or Min-Max scaling were not applicable. The model was expected to learn meaningful representations directly from the token sequences, with variance in scale inherently captured through embeddings and temporal context.

**References**

[1] Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016. https://colinraffel.com/projects/lmd/

[2] Jan Melechovsky, Abhinaba Roy, Dorien Herremans. "MidiCaps: A Large-Scale MIDI Dataset with Text Captions." arXiv preprint arXiv:2306.02541, 2023. https://arxiv.org/abs/2306.02541
