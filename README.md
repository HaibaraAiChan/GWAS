# GWAS  (MLP)

DataSet | func | patients num | SNPs num | p-value
------------ | ------------- | ------------- | ------------- | -------------
michailidu | train | 250,000 | 51,000 | YES
hunter     | test | 9,000 | ? | NO  

* **training dataset** 

		
    ```  
    1) ensembl: Protein name that contains the SNP. 
    2) SNP_ID: SNP name
    3) Position: the position of SNP in the protein
    4) wt_codon: the wildtype (original) sequence in DNA
    5) mu_codon: the mutant sequence in DNA
    6) wildtype: the original amino acid in the protein
    7) wildtype_value: the probability this amino acid is present in this protein depending on different organisms. (feature)
    8) Blosum62_wt: the score of staying the same and not having a mutation dependping on BLOSUM62 matrix. (feature)
    9) mutant: the mutant amino acid in the protein
    10) mutant_value: the probability this mutant amino acid is present in this protein depending on different organisms (feature)
    11) Blosum62_mu:  the score of changing the amino acid and have this mutation dependping on BLOSUM62 matrix. (feature)
    12) pdb: the crystal structure similar to the protein after aligning.
    13) disorder: detecting if this position take a specific fold and structure or not. (feature)
    14) confidence: the confidence of disorder or order of this position. (feature)
    15) pdb_aa: the amino acid in the crystal structure
    16) pdb_position: the position of the SNP in the crystal structure.
    17) core: does the SNP present in the core of the protein (feature)
    18) NIS (Non interacting surface): does the SNP present in the surface of the protein but not in the interface (feature)
    19) Interface: does the SNP present in the Protei-protein interaction interface (feature)
    20) HBO (Hydrogen bonding): does the SNP affect hydrogen bonding interaction (feature)
    21) SBR (Salt bridge): does the SNP affect salt bridge interaction (feature)
    22) Aromatic interaction: does the SNP affect aromatic interaction (feature)
    23) Hydrophobic Interaction: does the SNP affect hydrophobic interaction (feature)
    24) helix: 2ry structure of the protein (feature)
    25) coil: 2ry structure of the protein (feature)
    26) sheet: 2ry structure of the protein (feature)
    27) Entropy: the degree of freedom for this SNP (feature)
    28) P-value  
    ```
    
* **test dataset**   
	- [ ]  hunter  
	
* **MLP model:**  
		**input:**        
		**output:**  p-value: alpha   

# Target   
* **Predict the p-value of hunter dataset**
* **Is that possible to predict the p-value of a given dataset(random size)**

