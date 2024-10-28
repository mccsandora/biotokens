
import Bio
from Bio import SeqIO


cds_file_path = ""
ncrna_file_path = ""

#count num of sequences and their lengths in the fasta file
def verify_fasta_structure(file_path):
    sequences = []
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(len(record.seq))
            #return total sequences and lenths of first 50 for verification 
        return len(sequences), sequences[:50] 
    except Exception as e:
        return f"Error reading {file_path}: {e}"

cds_structure = verify_fasta_structure(cds_file_path)
ncrna_structure = verify_fasta_structure(ncrna_file_path)

print("cds struct:", cds_structure)
print("ncrna struct:", ncrna_structure)