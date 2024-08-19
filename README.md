# videowiz
This is a repo that is catered to my needs in video making.
The features include:
1. fetch images from various legitimate sources (wikimedia etc)
2. speech to text
3. video to audio

I intend to use this as a tool to collect data/images for the video making process.
docker build -t videowiz_env .

docker run -it --rm -v "$(pwd)":/app videowiz_env