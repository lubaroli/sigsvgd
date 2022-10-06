include Makefile.needed

HOST=https://raw.githubusercontent.com/soraxas/__file-store__/master
DATA_DIR=robodata
# DATASET=001_dataset.csv
# WEIGHTS=001_continuous-occmap-weight.ckpt

# WEIGHTS=$(addsuffix _continuous-occmap-weight.ckpt, $(TAGS))
# DATASETS=$(addsuffix _dataset.csv, $(TAGS))
# REQUESTS=$(addsuffix _request0001.yaml, $(TAGS))
# SCENES=$(addsuffix .yaml, $(TAGS))

# all:  $(addprefix $(DATA_DIR)/, $(DATASETS) $(WEIGHTS) $(REQUESTS) $(SCENES))
# all:  
# 	echo $(addprefix $(DATA_DIR)/, $(NEEDED))

all:  $(addprefix $(DATA_DIR)/, $(NEEDED))


$(DATA_DIR)/%:
	mkdir -p "$(DATA_DIR)"
	# wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/kitchen/$*"
	wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/$*"


