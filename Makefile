# external:
# 	mkdir external

external/pybullet-planning:
	git clone https://github.com/caelan/pybullet-planning external/pybullet-planning
	cd external/pybullet-planning && git submodule update --init --recursive -q

