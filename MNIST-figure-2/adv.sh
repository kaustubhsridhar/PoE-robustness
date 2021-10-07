while getopts f: option
do
case "${option}"
in
f) FOLDER=${OPTARG};;
esac
done

mkdir $FOLDER
cd $FOLDER
mkdir 0000
mkdir 0001
mkdir 0002
mkdir 0003
mkdir 0004
mkdir 0005
mkdir 0006
mkdir 0007
mkdir 0008
mkdir 0009