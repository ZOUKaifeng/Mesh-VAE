import argparse
import json
import os

parser = argparse.ArgumentParser( description = 'Analyse inference results', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument( 'path', help = 'inference path' )
parser.add_argument( "-p", "--prediction", help="Analyse sex prediction results", action="store_true" )
parser.add_argument( "-e", "--error", help="List max reconstruction errors", action="store_true" )
parser.add_argument( "-v", "--verbose", help="verbose", action="store_true" )
parser.add_argument( "-f", "--folds", help="number of folds", type = int )
args = parser.parse_args()

def report( path, args ):
	with open( os.path.join( path, "inference.json" ), 'r') as f:
		data = json.load( f )

	individuals = []
	numberOfPredictions = 0
	numberOfWrongPredictions = 0

	for file in data:
		data[ file ][ "file" ] = file
		individual = data[ file ]
		individuals.append( individual )
		numberOfPredictions += 1

		if args.prediction :
			sex = file.split( "_" )[ 1 ]
			if sex == "f" : sex = 0
			else : sex = 1

			if sex != individual[ "sex" ] :
				numberOfWrongPredictions += 1;
				if not args.verbose :continue
				print( file + ": wrong prediction" )
				if not "reconstruction_error" in individual : continue
				error = individual[ "reconstruction_error" ]
				print( "reconstruction error : max= " + str( error[ "max" ] ) +", mean= " + str( error[ "mean" ] ) )

	print( path, ":", str( numberOfPredictions ) + " predictions" )
	if args.prediction :
		accuracy = 100 - ( 100 * numberOfWrongPredictions / numberOfPredictions  )
		print( "{} wrong predictions. Accuracy : {:.2f}%)".format( numberOfWrongPredictions, accuracy ) )

	if args.error :
		print( "Sorted max errors:" )
		individuals.sort( key=lambda i: i[ "reconstruction_error" ][ "max" ] )
		for individual in individuals :
			print( individual[ "file" ] + " : " + str( individual[ "reconstruction_error" ][ "max" ] ) )

if args.folds:
	for fold in range( 1, args.folds + 1 ):
		report( os.path.join( args.path, str( fold ) ), args )
else:
	report( args.path, args )
