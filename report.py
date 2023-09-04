import argparse
import json

parser = argparse.ArgumentParser( description = 'Analyse inference results', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument( 'JSONFile', help = 'JSON inference results file' )
parser.add_argument( "-p", help="Analyse sex prediction results", action="store_true" )
parser.add_argument( "-e", help="List max reconstruction errors", action="store_true" )
args = parser.parse_args()

with open( args.JSONFile, 'r') as f:
  data = json.load( f )

individuals = []

numberOfPredictions = 0
numberOfWrongPredictions = 0

for file in data:
	data[ file ][ "file" ] = file
	individual = data[ file ]
	individuals.append( individual )
	numberOfPredictions += 1

	if args.p :
		sex = file.split( "_" )[ 1 ]
		if sex == "f" : sex = 0
		else : sex = 1

		if sex != individual[ "sex" ] :
			numberOfWrongPredictions += 1;
			print( file + ": wrong prediction" )
			error = individual[ "reconstruction_error" ]
			print( "reconstruction error : max= " + str( error[ "max" ] ) +", mean= " + str( error[ "mean" ] ) )

print( str( numberOfPredictions ) + " predictions" )
if args.p :
	accuracy = 100 - ( 100 * numberOfWrongPredictions / numberOfPredictions  )
	print( "{} wrong predictions. Accuracy : {:.2f}%)".format( numberOfWrongPredictions, accuracy ) )

if args.e :
	print( "Sorted max errors:" )
	individuals.sort( key=lambda i: i[ "reconstruction_error" ][ "max" ] )
	for individual in individuals :
		print( individual[ "file" ] + " : " + str( individual[ "reconstruction_error" ][ "max" ] ) )
