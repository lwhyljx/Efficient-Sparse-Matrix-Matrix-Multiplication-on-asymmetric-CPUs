#! /bin/bash
for((threadnum=1;threadnum<=8;threadnum=threadnum*2))
do
	for((M2=1;M2<=64;M2=M2*2))
	do
		for((K2=1;K2<=1;K2=K2*2))
		do
			for((N2=1;N2*threadnum<=256;N2=N2*2))
			do
				python3 SpMM_codegen.py --sparsity 80 --M ${M2} --K ${K2} --N ${N2} --numthread ${threadnum}
				cd codehub
				clang++  _generated_cpu_embed_sparse_matrix_64_128_256_80_${M2}_${N2}_${threadnum}.cpp -o a -lpthread -O3 -DNDEBUG
				./a
				cd ..
			done
		done
	done
done
