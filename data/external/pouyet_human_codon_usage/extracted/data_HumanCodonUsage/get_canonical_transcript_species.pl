# Ensembl API
# For each gene ID, retrieves the ID of the canonical transcript from Ensembl, 
# Run the perl script: perl get_canonical_transcript_species.pl geneID_file Human > out_file

use Bio::EnsEMBL::Registry;
use Bio::SeqIO;
use Data::Dumper;

# Check that there are 2 arguments: file of geneIDs + species name
unless ($ARGV[1]) {
      print STDERR "Usage: get_canonical_transcript_species GeneID_file species_name\n\n";
      exit;	
	}

# Open the list of gene IDs
unless ( open(file_list, $ARGV[0]) ) {
    print STDERR "EXIT: file not found: $ARGV[0] ...\n\n";
    exit;
}




# Use ensembl API
my $r = "Bio::EnsEMBL::Registry";
# Connect to local database: ensembl v75
$r->load_registry_from_db(-host => 'ensembldb.ensembl.org', -user => 'anonymous',-port => "5306", -verbose => "0");
# use gene_adaptor to get informations on genes
my $gene_adaptor=$r->get_adaptor($ARGV[1] ,"core", "Gene");
# Then, look at the list of gene :
# for each gene =


print "GeneID\tCanonicalTranscriptID\n";


foreach my$line (<file_list>) {
	if ( $line =~ /^\s*$/ ){
    	next;
    }
    else {
    	my @line_content = split(" ", $line);
	my $id = $line_content[0];

	# get the object corresponding to identification number
	my $gene = eval{$gene_adaptor->fetch_by_stable_id($id)};
	
	unless($gene){print STDERR "WARNING: ".$id.": GENE_ID NOT FOUND\n"; print $id."\tNA\n"; next;}; # print error and go to next line
	
	# get the canonical transcript
	my $canonical_transcript=$gene->canonical_transcript();
	# save the id
	my $canonical_id = $canonical_transcript->stable_id();
	print $id."\t".$canonical_id."\n";



	    }
}
close file_list;

