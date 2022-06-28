HOST=https://raw.githubusercontent.com/soraxas/__file-store__/master
DATA_DIR=robodata
DATASET=001_dataset.csv
WEIGHTS=001_continuous-occmap-weight.ckpt


all:  $(addprefix $(DATA_DIR)/, $(DATASET) $(WEIGHTS))


$(DATA_DIR)/%:
	mkdir -p "$(DATA_DIR)"
	wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/kitchen/$*"


