class Fasta():
    def __init__(self, fasta:str):
        self.fasta = fasta
        self._read_fasta()

    def _read_fasta(self):
        self.fasta_d = {}
        with open(self.fasta, 'r') as fasta:
            for line in fasta:
                if line.startswith(">"):
                    name = line.replace("\n", "")
                    self.fasta_d[name] = ''
                else:
                    self.fasta_d[name] += line.replace("\n", "")
    
    def split_fasta(self):
        for name, seq in self.fasta_d.items():
            filename = name[1:].split()[0].split("/")[0]+".fasta"
            with open(filename, 'w+') as ofile:
                ofile.write(name+"\n"+seq)

if __name__ == '__main__':
    fasta = Fasta('seed.fa')
    fasta.split_fasta()