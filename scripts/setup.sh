# scp -i ~/.ssh/burner-west2.pem -r data something@something.compute.amazonaws.com:.
# git clone git@github.com:sdsunjay/nlp_product_reviews.git
echo "cd ~"
cd ~
git config --global user.name "Sunjay Dhama"
read -p 'Email Address: ' email
git config --global user.email $email
ssh-keygen -t rsa -b 4096 -C $email
cat /home/ubuntu/.ssh/id_rsa.pub
echo "mkdir tmp"
mkdir tmp
echo "export TMPDIR=/home/ubuntu/tmp"
export TMPDIR=/home/ubuntu/tmp
echo "sudo apt update && sudo apt upgrade -y"
sudo apt update && sudo apt upgrade -y
echo "sudo apt-get install python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ virtualenv"
sudo apt-get install python3-dev python3-pip libxml2-dev libxslt1-dev zlib1g-dev g++ virtualenv
echo "sudo apt autoremove -y"
sudo apt autoremove -y
read -p "Press enter to continue"
echo "curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py"
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
echo "sudo python3 get-pip.py"
sudo python3 get-pip.py
echo "virtualenv -p python3 nlp_product_reviews"
virtualenv -p python3 nlp_product_reviews
echo "source nlp_product_reviews/bin/activate"
source nlp_product_reviews/bin/activate
echo "cd nlp_product_reviews"
cd nlp_product_reviews
echo "pip3 install -r requirements.txt"
pip3 install -r requirements.txt
echo "scp the data to me!"
echo "scp -i ~/.ssh/your_key.pem -r data something@something.compute.amazonaws.com:."
echo "don't forget to move data into nlp_product_reviews"
echo "mv data nlp_product_reviews/."
read -p "Press enter to continue"
echo "python preprocess.py"
python preprocess.py
