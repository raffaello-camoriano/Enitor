#!/usr/bin/gawk -f

#this script will do a binary expansion of the attributes of a clean adult database
#i denote a clean database, one without unknown attributes and without 'flnwgt' field

BEGIN {FS=","; OFS=","; IGNORECASE=1
			#print header information: field names

	H1 = "Age";
	H2 = "Private,Self-emp-not-inc,Self-emp-inc,Federal-gov,Local-gov,State-gov,Without-pay,Never-worked";
	H3 = "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool";
	H4 = "education-num";
	H5 = "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse";
	H6 = "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces";
	H7 = "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried";
	H8 = "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black";
	H9 = "Female, Male";
	H10= "Capital-gain";
	H11= "Capital-loss";
	H12= "Hours-per-week";
	H13= "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands";
	H14= "<=50K, >50K";


#			print "Age, Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked, Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool, Education-num, Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse, Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces, Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black, Female, Male, Capital-gain, Capital-loss, Hours-per-week, United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands, <=50K, >50K";
	
print H1, H2, H3, H4, H5, H6, H7, H8, H9, H10, H11, H12, H13, H14
		
	  }
{ 
	#age is continuous
	
	# $2 workclass attribute: 
	# Private,Self-emp-not-inc,Self-emp-inc,Federal-gov,Local-gov,State-gov,Without-pay,Never-worked	
	switch($2) {
		case /private/: $2=			" 1, 0, 0, 0, 0, 0, 0, 0"; break;
		case /self-emp-not-inc/: $2=" 0, 1, 0, 0, 0, 0, 0, 0"; break;
		case /self-emp-inc/: $2=	" 0, 0, 1, 0, 0, 0, 0, 0"; break;
		case /federal-gov/: $2=		" 0, 0, 0, 1, 0, 0, 0, 0"; break;
		case /local-gov/: $2=		" 0, 0, 0, 0, 1, 0, 0, 0"; break;
		case /state-gov/: $2=		" 0, 0, 0, 0, 0, 1, 0, 0"; break;
		case /without-pay/: $2=		" 0, 0, 0, 0, 0, 0, 1, 0"; break;
		case /never-worked/: $2=	" 0, 0, 0, 0, 0, 0, 0, 1"; break;
		default:
	}
	
	#education atribute
	# Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool
	switch($3) {
	
		case /bachelors/: $3=		" 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /some-college/: $3= 	" 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /11th/: $3=			" 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /hs-grad/: $3= 		" 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Prof-school/: $3= 	" 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break; 
		case /Assoc-acdm/: $3=		" 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;  
		case /Assoc-voc/: $3=		" 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break; 
		case /9th/: $3=				" 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0"; break; 
		case /7th-8th/: $3=			" 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0"; break; 
		case /12th/: $3=			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0"; break;  
		case /Masters/: $3=			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0"; break;  
		case /1st-4th/: $3=			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0"; break;  
		case /10th/: $3=			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0"; break;  
		case /Doctorate/: $3=		" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0"; break;  
		case /5th-6th/: $3=			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0"; break;  
		case /Preschool/: $3=		" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1"; break; 
		default:
	}
	
	# $4 education-num is continuous
	
	# $5 attribute marital-status
	# Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse
	switch($5) {	
		case /Married-civ-spouse/: $5= 		" 1, 0, 0, 0, 0, 0, 0"; break;
		case /Divorced/: $5= 				" 0, 1, 0, 0, 0, 0, 0"; break;
		case /Never-married/: $5= 			" 0, 0, 1, 0, 0, 0, 0"; break;
		case /Separated/: $5= 				" 0, 0, 0, 1, 0, 0, 0"; break;
		case /Widowed/: $5= 				" 0, 0, 0, 0, 1, 0, 0"; break;
		case /Married-spouse-absent/: $5= 	" 0, 0, 0, 0, 0, 1, 0"; break;
		case /Married-AF-spouse/: $5=		" 0, 0, 0, 0, 0, 0, 1"; break;
		default:
	}
	
	# %6 attribute occupation
	# Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces
	switch($6) {	
		case /Tech-support/: $6= 		" 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Craft-repair/: $6= 		" 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Other-service/: $6= 		" 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Sales/: $6= 				" 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;	
		case /Exec-managerial/: $6= 	" 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Prof-specialty/: $6= 		" 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Handlers-cleaners/: $6= 	" 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Machine-op-inspct/: $6= 	" 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0"; break;
		case /Adm-clerical/: $6= 		" 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0"; break;
		case /Farming-fishing/: $6= 	" 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0"; break;
		case /Transport-moving/: $6= 	" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0"; break;
		case /Priv-house-serv/: $6= 	" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0"; break;
		case /Protective-serv/: $6= 	" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0"; break;
		case /Armed-Forces/: $6=		" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1"; break;
		default:
	}
	
	# $7 attribute relationship
	# Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried
	switch($7) {
		case /Wife/: $7=			" 1, 0, 0, 0, 0, 0"; break; 
		case /Own-child/: $7= 		" 0, 1, 0, 0, 0, 0"; break;
		case /Husband/: $7= 		" 0, 0, 1, 0, 0, 0"; break;
		case /Not-in-family/: $7= 	" 0, 0, 0, 1, 0, 0"; break;
		case /Other-relative/: $7= 	" 0, 0, 0, 0, 1, 0"; break;
		case /Unmarried/: $7=		" 0, 0, 0, 0, 0, 1"; break;
		default:	
	}
	
	# $8 attribute race
	# White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black
	switch($8) {
		case /White/: $8= 				" 1, 0, 0, 0, 0"; break;
		case /Asian-Pac-Islander/: $8= 	" 0, 1, 0, 0, 0"; break;
		case /Amer-Indian-Eskimo/: $8= 	" 0, 0, 1, 0, 0"; break;
		case /Other/: $8= 				" 0, 0, 0, 1, 0"; break;
		case /Black/: $8=				" 0, 0, 0, 0, 1"; break;
		default:
	}
	
	#sex attribute, use explicit comparison because regex expression fails in this case
	if($9 == " Male")
		$9 = " 0, 1";
	else if($9 == " Female")
		$9 = " 1, 0";
	
	
	# $10 attribute capital-gain is continuos
	
	# $11 attribute capital-loss is continuos
	
	# $12 attribute hours-per-week is continuos
	
	# $13 attribute native-country
	# United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
	switch($13) {
		case /United-States/: $13= 				" 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Cambodia/: $13= 					" 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /England/: $13= 					" 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Puerto-Rico/: $13= 				" 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Canada/: $13= 					" 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Germany/: $13= 					" 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Outlying-US(Guam-USVI-etc)/: $13= " 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /India/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Japan/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Greece/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /South/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /China/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Cuba/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Iran/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Honduras/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Philippines/: $13= 				" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Italy/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Poland/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Jamaica/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Vietnam/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;	
		case /Mexico/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Portugal/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Ireland/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /France/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Dominican-Republic/: $13= 		" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Laos/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Ecuador/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Taiwan/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Haiti/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Columbia/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Hungary/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Guatemala/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Nicaragua/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Scotland/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0"; break;
		case /Thailand/: $13= 					" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0"; break;
		case /Yugoslavia/: $13= 				" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0"; break;
		case /El-Salvador/: $13= 				" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0"; break;
		case /Trinadad&Tobago/: $13= 			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0"; break;
		case /Peru/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0"; break;
		case /Hong/: $13= 						" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0"; break;
		case /Holand-Netherlands/: $13=			" 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1"; break;
		default:
	}
	
	# $14 attribute class >=50 K
	# >50K, <=50K
	switch($14) {
		case />50K/: $14= " 0, 1"; break;
		case /<=50K/:$14= " 1, 0"; break;
		default:
	}
	

#print $0	
#print $1, $2, $3, $5, $7, $8, $9, $14
print $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14

}
		
