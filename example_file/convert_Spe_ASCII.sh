#!/bin/bash
#script to comment out Maestro header lines from Spe format file and
#copy data lines to output
# Carl Wheldon
# University of Birmingham
# Retrieved from http://www.np.ph.bham.ac.uk/research_resources/programs/spec_conv/maestro.bash

for i in *.Spe;
    do
#       Print input and output filenames to screen
        echo " ";
        echo "Maestro ascii spectrum file:" $i;
        OUTFL="`basename "$i" .Spe`.txt";
        echo "Output file name:" $OUTFL;
#       copy the 12 header lines to output file with hash in front (i.e. comments)
        gawk '{if(NR<=12) {print "#" $0}}' $i > $OUTFL;
#        gawk '{if(NR==12) {print "#" $0}}' $i >> $OUTFL;
#       extract number of data lines. Need printf to get number as integer
        LNCNT=`gawk '{if(NR==12) {printf("%d",$2)}}' $i`;
#       Number starts from zero so add 1
        let "LNCNT += 1";
        echo "Number of channels:" $LNCNT;
#       add 12 head files to get the right line number
        let "LNCNT += 12";
#        echo "End line number:" $LNCNT;
        gawk -v AWKLNCNT="$LNCNT" '{if(NR>12 && NR<=(AWKLNCNT)) {printf("%d\t%d\n",NR-13,$1)}}' $i >> $OUTFL;
        echo "Done";
        echo " ";
    done