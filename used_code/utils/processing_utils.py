from collections import defaultdict
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def read_gffs(gff_file_path, genome_file_path, output_file_path):
    # Load genome sequence
    genome_dict = SeqIO.to_dict(SeqIO.parse(genome_file_path, "fasta"))

    # Dictionary to store CDS features by their Parent ID
    cds_by_parent = defaultdict(list)

    # Parse the GFF file
    with open(gff_file_path) as file:
        for line in file:
            if not line.startswith("#"):
                parts = line.strip().split('\t')
                if parts[2] == "CDS":
                    # Extract relevant data
                    seq_id = parts[0]
                    start = int(parts[3]) - 1  # GFF is 1-based, converting to 0-based
                    end = int(parts[4])
                    strand = parts[6]
                    attributes = parts[8]

                    # Extract Parent ID
                    parent_id = None
                    for attribute in attributes.split(';'):
                        if attribute.startswith("Parent="):
                            parent_id = attribute.split('=')[1]
                            break

                    if parent_id:
                        # Append CDS feature details
                        cds_by_parent[parent_id].append((seq_id, start, end, strand))

    # Function to join CDS sequences
    def join_cds(cds_list):
        cds_list.sort(key=lambda x: x[1])  # Sort by start position
        joined_seq = Seq('')
        sequence_id = ''
        for seq_id, start, end, strand in cds_list:
            seq = genome_dict[seq_id].seq[start:end]
            if strand == '-':
                seq = seq.reverse_complement()
            joined_seq += seq
            sequence_id = seq_id
        return joined_seq, sequence_id

    # Join CDS for each Parent ID
    cds_records = []
    for parent_id, cds_list in cds_by_parent.items():
        joined_cds_seq, seq_id = join_cds(cds_list)
        cds_record = SeqRecord(joined_cds_seq, id=seq_id, description="")
        cds_records.append(cds_record)

    # Save to a FASTA file

    SeqIO.write(cds_records, output_file_path, "fasta")
    return cds_records