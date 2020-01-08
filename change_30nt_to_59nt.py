import pandas as pd

rbs = 'AACAGAGGAGA'
start_codon = 'ATG'

# Make function to generate reverse compliment of the DNA strand
def make_rev_complement(string):
    new_str = ''
    for s in string:
        char = ''
        if s == 'A':
            char = 'T'
        elif s == 'T':
            char = 'A'
        elif s == 'C':
            char = 'G'
        elif s == 'G':
            char = 'C'
        else:
            print('UH OH! Character not A, T, C, or G')
        new_str += char
    new_str = new_str[::-1]
    return new_str

# Make function to check for stop codons
def check_for_stop(toehold): 
    stop_codons = ['TAG', 'TAA', 'TGA']
    location_of_start = 47
    search1 = toehold.find(stop_codons[0]) == location_of_start
    search2 = toehold.find(stop_codons[1]) == location_of_start
    search3 = toehold.find(stop_codons[2]) == location_of_start
    return (search1 | search2  | search3)

# Make function to actually turn trigger into toehold
def turn_switch_to_toehold(switch):
    stem1 = make_rev_complement(switch[24:30])
    stem2 = make_rev_complement(switch[12:21])
    toehold = switch + rbs + stem1 + start_codon + stem2
    return toehold

# main function
def main(switches):
    toeholds = [turn_switch_to_toehold(x) for x in switches]
    #print('Total of ' + str(len(toeholds)) + ' number of switches.')
    no_stop = [x for x in no_start if not check_for_stop(x)]
    #print('After checking for stop codons, total of ' + str(len(no_stop)) + ' number of switches.')

    all_new_toeholds = pd.DataFrame(no_stop)
    return no_stop

