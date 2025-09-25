#!/usr/bin/env bash

testdir=$1
pushd $testdir

iree-compile \
	--iree-hal-target-backends=llvm-cpu \
	test.mlir \
	-o file.vmfb 2> /dev/null

inputs=$(ls input*.npy | sort --human-numeric-sort |  awk '{print "--input=@" $1}')
results=$(ls result*.npy | sort --human-numeric-sort |  awk '{print "--expected_output=@" $1}')

iree-run-module \
	--module=file.vmfb \
	--device=local-task \
	$inputs \
	$results

rm file.vmfb 2> /dev/null
popd 
