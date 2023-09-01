import argparse
from config_parser import read_config
import json
import math
from humanfriendly import format_timespan
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser( description = 'Plot training history', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument( 'JSONFile', help = 'JSON inference training history file' )
parser.add_argument( "-d", "--display", help="Display plot on screen", action="store_true" )
parser.add_argument( "-o", "--output", help="output file" )
parser.add_argument( "-c", "--config", help="model and training config file" )
args = parser.parse_args()

with open( args.JSONFile, 'r') as f : data = json.load( f )
types = [ "training", "validation" ]
lossTypes = list( data[ 0 ][ types[ 1 ] ].keys() )
epochs = list( map( lambda e : e[ 'epoch' ], data ) )

figure = plt.figure( figsize = ( 18, 10 ) )
figure.suptitle( args.JSONFile, fontsize=16 )

width = 3#math.ceil( len( lossTypes ) / 2 )
pos = width * 100 + width * 10;

for loss in lossTypes:
	pos += 1
	lossTxt = " ".join( loss.split( "_" ) )
	ax = figure.add_subplot( pos )
	ax.set_xlabel( 'epoch' )
	ax.set_ylabel( lossTxt )
	ax.set_xlim( 0, epochs[ -1 ] )

	for type in types:
		if loss not in data[ 0 ][ type ] : continue
		values = list( map( lambda e : e[ type ][ loss ], data ) )
		ax.plot( epochs, values, label = type)

	ax.legend( title = lossTxt, loc = 'center right')

duration = data[ -1 ][ 'begin' ] - data[ 0 ][ 'begin' ] + data[ -1 ][ "duration" ]
text = "Total training time : " + format_timespan( math.ceil( duration ) )
if "test" in data[ -1 ] :
	text += "\ntest : " + json.dumps( data[ -1 ][ "test" ] )
if args.config : text += "\nConfig : " + json.dumps( read_config( args.config ) )
figure.text( 0.1, 0.15, text, wrap = True )

if args.display : plt.show()
if args.output : plt.savefig( args.output )
