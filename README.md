# Local automatic speech recognition service

## Architecture
In this version, each service runs in a Docker container. The gateway does not run in a container but uses the docker package to start and stop containers.
### Rationale:
Limited vRAM. Originally, each transcriber (local Whisper, Faster Whisper) was supposed to be an object (subclass of a base class) to be loaded and unloaded from memory. However, I could not install Faster Whisper with the original Whisper in the same venv, thus different Docker containers for different ASR engines.

Folder structure in Docker containers will be the same as in the project: starting from the 'app' folder at the top to prevent PyCharm's import warnings, but only the required files will be copied into a container. (The _app_ folder is a subfolder of the _project_ folder in a  container so that the _app_ folder recognized as a package.)

Due to limited memory, only one transcription container is allowed to run at a time so they all use the same port (3001).

## Docker
Build context for all transcribers is the app folder, e.g.:
~/MyCode/PyCharm/ASR/app$ docker build -t faster_whisper:0 -f ./transcribers/faster_whisper/Dockerfile .


## TODO
- Correct data submission from index.html
- Do not unload model if it is None
- Meaningful data validation response from a container
- Container for local Whisper
- Log level from .env inside containers
- Containers for NLP services
- Get a full list of requirements out of the Faster Whisper container
- Find out why the service only starts working with the second request
