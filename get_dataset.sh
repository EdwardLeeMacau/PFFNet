# Dataset of raw images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e5OhBF4f3sNCxEq-N8W55gzXsQ04qcZi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1e5OhBF4f3sNCxEq-N8W55gzXsQ04qcZi" -O ntire2018_raw.zip && rm -rf /tmp/cookies.txt

unzip ./ntire2018_raw.zip -d ntire2018_raw

rm ./ntire2018_raw.zip

# Dataset of GAN images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1upD_2nZJXqYOrlhfLuiM4xhrYA05dc-5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1upD_2nZJXqYOrlhfLuiM4xhrYA05dc-5" -O ntire2018_gan.zip && rm -rf /tmp/cookies.txt

unzip ./ntire2018_gan.zip -d ntire2018_gan

rm ./ntire2018_gan.zip

# Dataset of 640 images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nol9VPdJH93Y_3E2qZPkeCMSzTfVokM0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nol9VPdJH93Y_3E2qZPkeCMSzTfVokM0" -O ntire2018_640.zip && rm -rf /tmp/cookies.txt

unzip ./ntire2018_640.zip -d ntire2018_640

rm ./ntire2018_640.zip