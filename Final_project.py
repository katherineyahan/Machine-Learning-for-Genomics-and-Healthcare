# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 22:04:00 2023

@author: 张雅涵
"""
ldm = mgp.LDMatrix.from_path("./chr_22")
# print the number of SNPs:
print(ldm.n_snps)
# Convert to sparse matrix format:
cst_mat = ldm.to_csr_matrix()
a=ldm.snps