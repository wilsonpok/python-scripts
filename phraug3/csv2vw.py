# Python 3 version of csv2vw.py from https://github.com/zygmuntz/phraug2

'Convert CSV file to Vowpal Wabbit format.'
'Allows mixing of categorical and numerical data'

import sys
import csv
import argparse

def clean( item ):
	return "".join( item.split()).replace( "|", "" ).replace( ":", "" )

def handle_label( label ):
	try:
		label = float( label )

		if label == 0.0:
			if args.convert_zeros:
				label = "-1"
			else:
				label = "0"
		elif label == 1.0:
			label = '1'

	except:
		if label == '':
			print("WARNING: a label is ''")
		else:
			print("WARNING: a label is '{}', setting to ''".format( label ))
			label = ''

	return label

def construct_line( label, line ):

	new_line = []
	new_line.append( "{} |n".format( handle_label( label )))

	# the rest

	for i, item in enumerate( line ):

		if i in ignore_columns_dict:
			continue

		if args.categorical:
			# 1-based indexing here
			new_item = "c{}_{}".format( i + 1, clean( item ))

		else:
			categorical = False
			try:
				item_float = float( item )
				if item_float == 0.0:
					continue    # sparse format
			except ValueError:
				if item:
					categorical = True
				else:
					continue

			if categorical:
				new_item =  "c{}_{}".format( i + 1, clean( item ))
			else:
				new_item = "{}:{}".format( i + 1, item )


		new_line.append( new_item )

	new_line = " ".join( new_line )
	new_line += "\n"
	return new_line

# ---

parser = argparse.ArgumentParser( description = 'Convert CSV file to Vowpal Wabbit format.' )
parser.add_argument( "input_file", help = "path to csv input file" )
parser.add_argument( "output_file", help = "path to output file" )

parser.add_argument( "-s", "--skip_headers", action = "store_true",
	help = "use this option if there are headers in the file - default false" )

parser.add_argument( "-l", "--label_index", type = int, default = 0,
	help = "index of label column (default 0, use -1 if there are no labels)")

parser.add_argument( "-z", "--convert_zeros", action = 'store_true', default = False,
	help = "convert labels for binary classification from 0 to -1" )

parser.add_argument( "-i", "--ignore_columns",
	help = "zero-based index(es) of columns to ignore, for example 0 or 3 or 3,4,5 (no spaces in between)" )

parser.add_argument( "-c", "--categorical", action = 'store_true',
	help = "treat all columns as categorical" )

parser.add_argument( "-n", "--print_counter", type = int, default = 10000,
	help = "print counter every _ examples (default 10000)" )

parser.add_argument( "-d", "--delimiter", default = ",",
	help = "delimiter used to separate columns" )

args = parser.parse_args()

###

ignore_columns = []

if args.ignore_columns:
	ignore_columns = args.ignore_columns.split( ',' )
	ignore_columns = map( int, ignore_columns )
	print("ignoring columns", ignore_columns)

	if args.label_index in ignore_columns:
		raise(ValueError, "You are not trying to ignore the label column, are you?")

	# correct for later popping the label
	if args.label_index >= 0:
		ignore_columns = map( lambda x: x - 1 if x > args.label_index else x, ignore_columns )

# a dictionary for faster 'in'
ignore_columns_dict = { x: 1 for x in ignore_columns }

###

i = open( args.input_file )
o = open( args.output_file, 'wb' )

reader = csv.reader( i, delimiter = args.delimiter )
if args.skip_headers:
	headers = next(reader)

n = 0

for line in reader:
	if args.label_index < 0:
		label = 1
	else:
		label = line.pop( args.label_index )

	new_line = construct_line( label, line )
	o.write( new_line.encode() )

	n += 1
	if n % args.print_counter == 0:
		print(n)


