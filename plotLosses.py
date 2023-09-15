import argparse
from config_parser import read_config
import json
import math
from humanfriendly import format_timespan
import matplotlib.pyplot as plt

def plotLosses( title, data, config = None ) :
	types = [ "training", "validation" ]
	lossTypes = list( data[ 0 ][ types[ 1 ] ].keys() )
	epochs = list( map( lambda e : e[ 'epoch' ], data ) )

	figure = plt.figure( figsize = ( 18, 10 ) )
	figure.suptitle( title, fontsize=16 )

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

		ax.legend( title = lossTxt )

	pos += 1
	ax = figure.add_subplot( pos )
	ax.set_xlabel( 'epoch' )
	ax.set_ylabel( "duration (s.)" )
	ax.set_xlim( 0, epochs[ -2 ] )
	durations = []
	x = []
	y = []
	nSaves = 0
	text = ""

	for epoch in epochs:
		if epoch < epochs[ -2 ]:
			duration = data[ epoch ][ "begin" ] - data[ epoch -1 ][ "begin" ]
		durations.append( duration )
		if "saved" in data[ epoch -1 ]:
			x.append( epoch )
			y.append( duration )
			nSaves += 1
			text = "Last model saved at epoch {}, {} total saves\n".format( epoch, nSaves )
	ax.plot( epochs, durations, label = "duration(s.)" )
	ax.plot( x, y, marker='*', ls='none', ms=10, label = "model saves")
	ax.legend()

	duration = data[ -1 ][ 'begin' ] - data[ 0 ][ 'begin' ] + data[ -1 ][ "duration" ]
	text += "Total training time : " + format_timespan( math.ceil( duration ) )
	if "test" in data[ -1 ] :
		text += "\ntest : " + json.dumps( data[ -1 ][ "test" ] )
	if config : text += "\nConfig : " + json.dumps( config )
	figure.text( 0.4, 0.15, text, wrap = True )
	return plt


if __name__ == '__main__':
	parser = argparse.ArgumentParser( description = 'Plot training history', formatter_class=argparse.ArgumentDefaultsHelpFormatter )
	parser.add_argument( 'JSONFile', help = 'JSON inference training history file' )
	parser.add_argument( "-d", "--display", help="Display plot on screen", action="store_true" )
	parser.add_argument( "-o", "--output", help="output file" )
	parser.add_argument( "-c", "--config", help="model and training config file" )
	args = parser.parse_args()

	with open( args.JSONFile, 'r') as f : data = json.load( f )
	config = None
	if args.config : config = read_config( args.config )
	plt = plotLosses( args.JSONFile, data, config )
	if args.display : plt.show()
	if args.output : plt.savefig( args.output )
