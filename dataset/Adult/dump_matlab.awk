#!/usr/bin/gawk -f

BEGIN {FS=","; 

print "function adult = create_adult_dataset()";

}
{
	if(NR==1) {
		print "adult.names = { \n";

		for (i=1; i<=NF; i++)
			printf("'%s';\n", $i)
		printf("\n") # CR at end of line
		
		print "};\n\n";

		print "adult.samples = [ "
	}
	else {
		for (i=1; i<=NF; i++)
			printf("%s ", $i)
		printf(";\n") # CR at end of line
	}
}

END { print "];"}


