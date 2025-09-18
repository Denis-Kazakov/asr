# Local automatic speech recognition service

__Purpose:__ create a speech-to-text service that can be deployed on a computer with limited vRAM but can use different speech-to-text engines and models and has other features such as transcript segmentation by sentences, hallucination and gap detection, etc. 

## Architecture
In this version, each service runs in a Docker container. The gateway does not run in a container but uses the docker package to start and stop containers.
Rationale: 
Originally, each transcriber (local Whisper, Faster Whisper) was supposed to be an object (subclass of a base class) to be loaded and unloaded from memory. However, I could not install Faster Whisper with the original Whisper in the same venv, thus different Docker containers for different ASR engines.

Folder structure in Docker containers will be the same as in the project: starting from the 'app' folder at the top to prevent PyCharm's import warnings, but only the required files will be copied into a container. (The _app_ folder is a subfolder of the _project_ folder in a  container so that the _app_ folder recognized as a package.)

Due to limited memory, only one transcription container is allowed to run at a time so they all use the same port (3001).

NLP models that can run on GPU (Stanza) are forced to run on CPU to minimize container loading/unloading as their workload is much smaller than speech recognition. Only one sentence tokenizer is allowed to run at a time to save RAM, so their ports are also the same (3002).

## Docker
Build context for all transcribers is the app folder, e.g.:
~/MyCode/PyCharm/ASR/app$ docker build -t faster_whisper:0 -f ./transcribers/faster_whisper/Dockerfile .

For local Whisper, start with a Pytorch image:
docker build -t local_whisper:0 -f ./transcribers/local_whisper/Dockerfile.torch .

For Stanza:
docker build -t stanza:0 -f ./nlp/transcript_segmentation/stanza/Dockerfile .


## TODO
- Graceful shutdown with container stopping
