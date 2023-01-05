bfd_db='/data00/alphafold_data_p2/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
uni_30='/data00/alphafold_data_p2/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
for seq in seeds/*;
do echo nohup hhblits -i $seq -o $seq.hhr -oa3m $seq.a3m -n 4 -d $bfd_db -d $uni_30 -cpu 1 \&;
done