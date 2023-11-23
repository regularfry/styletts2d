# StyleTTS 2 Daemon

This is a very naive reorganisation of the [StyleTTS 2](https://styletts2.github.io/)
[jupyter code](https://github.com/yl4579/StyleTTS2) to wrap it in a unix domain
socket listener.  This means the model is loaded once at startup.

I've only wrapped the LibriTTS model, mostly out of laziness.

## Installation

Courtesy of `eigenvalue` [over on HN](https://news.ycombinator.com/item?id=38338932),
these instructions should get you started on Linux:

```shell
git clone https://github.com/regularfry/styletts2d.git
cd styletts2d
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install wheel
pip install -r requirements.txt
pip install phonemizer librosa gdown
gdown "https://drive.google.com/uc?id=1jK_VV3TnGM9dkrIMsdQ_upov8FrIymr7"
7zz x Models.zip
rm Models.zip
pip install pickleshare nltk
python -c "import nltk; nltk.download('punkt')"
```

Then on linux:

```shell
sudo apt-get install -y espeak-ng
```

Or on OS X:

```shell
brew install espeak
```

## Usage

First, record a wav of yourself speaking for a few seconds.  Put it somewhere like `demo/reference_audio/me.wav`

Start the server:

```
source venv/bin/activate
python ./serve.py /tmp/tts.sock demo/reference_audio/me.wav
```

If you get an error that says `espeak not found. Please install espeak or specify the path to the espeak library with the -e option.` then you need to tell it where to find the espeak library.  On OS X, with an `espeak` from homebrew, that looks like this:

```shell
 $ brew info espeak | grep opt
/opt/homebrew/Cellar/espeak/1.48.04_1 (298 files, 3MB) *
 $ ls /opt/homebrew/Cellar/espeak/1.48.04_1/lib
libespeak.1.1.48.dylib  libespeak.1.dylib  libespeak.a  libespeak.dylib
 $ python ./serve.py /tmp/tts.sock demo/reference_audio/me.wav --espeak-path /opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib
```

Now, with the server running, you can open another terminal, and (in the same directory) run:

```shell
 $ python ./client.py /tmp/tts.sock "There will be... cake."
```

After a short delay, if all is well, you will hear tasty promises being made.

It will also accept piped input. This is equivalent:

```shell
 $ echo "There will be... cake." | python ./client.py /tmp/tts.sock
```

## Details

If you want to tweak the model settings (alpha, beta, and so on from the original code) you can
find them in the `Voice` class in `tts.py`. I'll probably expose them as CLI arguments when I get
round to it.

## Author

Original author as credited [here](https://github.com/yl4579/StyleTTS2/blob/main/LICENSE): Aaron (Yinghao) Li
Graceless slash and burn by me: [Alex Young](mailto:alex@blackkettle.org)