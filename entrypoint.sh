#!/bin/bash

action=$1
shift

[ ! -d "./cache" ] && ln -sf "$QA_CACHE_PATH" ./cache
[ ! -d "./data" ] && ln -sf "$QA_DATA_PATH" ./data

case $action in
	"bootstrap")
		python3 init_container.py
		python3 update_index.py
		python3 questionable.py
		;;
	"prepare")
		python3 init_container.py $@
		;;
	"index")
		python3 update_index.py $@
		;;
	"serve")
		python3 questionable.py $@
		;;
	*)
		cat HELP.md
esac
