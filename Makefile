HOST=https://raw.githubusercontent.com/soraxas/__file-store__/master
DATA_DIR=robodata
# DATASET=001_dataset.csv
# WEIGHTS=001_continuous-occmap-weight.ckpt

TAGS= bookshelf_small_panda-scene0001 bookshelf_tall_panda-scene0001 bookshelf_thin_panda-scene0001 box_panda-scene0001 cage_panda-scene0001 table_bars_panda-scene0001 table_pick_panda-scene0001 table_under_pick_panda-scene0001

WEIGHTS=$(addsuffix _continuous-occmap-weight.ckpt, $(TAGS))
DATASETS=$(addsuffix _dataset.csv, $(TAGS))
REQUESTS=$(addsuffix _request0001.yaml, $(TAGS))
SCENES=$(addsuffix .yaml, $(TAGS))

all:  $(addprefix $(DATA_DIR)/, $(DATASETS) $(WEIGHTS) $(REQUESTS) $(SCENES))


$(DATA_DIR)/%:
	mkdir -p "$(DATA_DIR)"
	# wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/kitchen/$*"
	wget  --directory-prefix="$(DATA_DIR)" "$(HOST)/stein/$*"


