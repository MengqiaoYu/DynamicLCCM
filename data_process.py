import csv
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger()

def remove_na(data, header):
    """
    remove the lines that contain 'NA'
    return the new dataset
    """
    index_to_del = []
    for i, item in enumerate(data):
        if 'NA' in item or '' in item:
            index_to_del.append(i)
            logger.debug("There is NA in this line:%s." %(item))
    for i in sorted(index_to_del, reverse=True):
        del data[i]
    logger.info("\tThere are %d lines in the cleaned dataset." %(len(data)))
    id_clean_lh = [item[header_lh.index('id')] for item in data]
    logger.info("\tThere are %d unqiue id (person) in the cleaned dataset." %(len(set(id_clean_lh))))
    return data

def extract_young_data(data, header, cutoff):
    """
    extract those people that contain full data points from age 20-31 (12 years)
    return a list of data # of lines must = # of people * 15 (years)
    note that there shouldn'pt be any 'NA'
    """
    id_curr = -1
    results = []
    result_curr = []
    count = 0

    for item in data:
        birthyear_curr = int(item[header.index('birthyear')])

        # Consider people born in [1946, 1987]
        if birthyear_curr >= cutoff or birthyear_curr <= 1945:
            continue

        if id_curr != item[header.index('id')]:
            if len(result_curr) == 2000 - cutoff: # 2018 - cutoff + 1 - 20 + 1
                results.extend(result_curr)
                count += 1 # count number of effective people
            result_curr = []
            id_curr = item[header.index('id')]

        # Consider the data in the age [20, 30]
        if int(item[header.index('year')]) - birthyear_curr <= 2019 - cutoff: # 2018 - cutoff + 1
            item.append(find_generation(birthyear_curr))
            result_curr.append(item)

    # Dataset summary
    logger.info("\tThere are %d lines in the cleaned young people version" %(len(results)))
    logger.info("\tThere are %d unqiue id (person)." %(count))
    assert count * (2000 - cutoff) == len(results), "Warning: inconsistent number of lines."

    # Save to file
    data_dir = "/Users/MengqiaoYu/Desktop/WholeTraveler/Data"
    logger.info("\tSave the new dataset to %s" %(data_dir))
    header.append("generation") # We have added another column: "generation".
    with open(data_dir + '/young_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)
    return results

def find_behavior_change(data_lh):
    """
    return a reformatted dataset that contains all the lines that contain a change,
    either travel behavior or life events.
    """
    prev_id = -1
    data_change = []

    for item in data_lh:
        try:
            curr_id = int(item[header_lh.index('id')])
        except Exception:
            logger.debug("Cannot find id in this line:%s." %(item))
        curr_numcars = int(item[header_lh.index('numcars')])
        curr_employ = int(item[header_lh.index('employ')])
        curr_youngchild = int(item[header_lh.index('youngchild')])
        curr_partner = int(item[header_lh.index('partner')])
        curr_public = int(item[header_lh.index('used_public')])
        curr_ridehail = int(item[header_lh.index('used_ridehail')])
        curr_own = int(item[header_lh.index('used_own')])
        curr_walkbike = int(item[header_lh.index('used_walkbike')])

        if curr_id == prev_id:
            numcars_change = curr_numcars - prev_numcars
            employ_change = curr_employ - prev_employ
            youngchild_change = curr_youngchild - prev_youngchild
            partner_change = curr_partner - prev_partner
            public_change = curr_public - prev_public
            ridehail_change = curr_ridehail - prev_ridehail
            own_change = curr_own - prev_own
            walkbike_change = curr_walkbike - prev_walkbike
            change_bool = [numcars_change,
                           int(item[header_lh.index('child')]),
                           int(item[header_lh.index('move')]),
                           int(item[header_lh.index('edu')]),
                           partner_change,
                           youngchild_change,
                           employ_change,
                           public_change,
                           ridehail_change,
                           own_change,
                           walkbike_change]
            if any(change_bool):
                # import pdb; pdb.set_trace()
                data_change.append([curr_id,
                                    item[header_lh.index('age')],
                                    item[header_lh.index('birthyear')]]
                                   + change_bool)
        prev_id = curr_id
        prev_numcars = curr_numcars
        prev_employ = curr_employ
        prev_youngchild = curr_youngchild
        prev_partner = curr_partner
        prev_public = curr_public
        prev_ridehail = curr_ridehail
        prev_own = curr_own
        prev_walkbike = curr_walkbike

    assert len(data_change) > 0, "Alert: no behavior change in this dataset!"
    header_change = ['id', 'age', 'birthyear', 'numcars', 'child', 'move', 'edu',
                     'partner', 'youngchild', 'employ', 'used_public',
                     'used_ridehail', 'used_own', 'used_walkbike']
    data_dir = "/Users/MengqiaoYu/Desktop/WholeTraveler/Data"
    logger.info("Save the behavior change file to %s" %(data_dir))
    with open(data_dir + '/behavior_change.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header_change)
        writer.writerows(data_change)
    return data_change

def find_generation(birthyear):
    """
    find the generation given an age (int)
    return an string denoting the name the generation

    """
    ### 20-25, 26-30, 31-35
    if birthyear >= 1980:
        return "millennial"
    elif birthyear >= 1966:
        return "genX"
    elif birthyear >= 1946:
        return "boomer"
    else:
        return "silent"

def save_model_data(filepath, num_year):
    """
    save all the long format longitudinal data into each single file.
    no return
    """
    data_all = []
    with open(filepath,'r') as inputfile:
        for row in csv.reader(inputfile):
            data_all.append(row)
    header = data_all[0]
    data_all = data_all[1:]
    for i in range(len(data_all)//num_year):
        data_ind = data_all[i * num_year: (i + 1) * num_year]
        with open('/Users/MengqiaoYu/Desktop/WholeTraveler/Data/model/' + str(i+1) + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_ind)


"""Load life history dataset. Nothing is done right now."""
### one person one year per line, 'lh' suffix represents lifehistory
logger.info("(I) Start loading two datasets. No data cleaning.")
rawdata_lh = []
with open("/Users/MengqiaoYu/Desktop/WholeTraveler/Data/clean_full_lifehistory_final_rawdata.csv", 'r') as input_lifehistory:
    for row in csv.reader(input_lifehistory):
        rawdata_lh.append(row)
header_lh = rawdata_lh[0]
rawdata_lh = rawdata_lh[1:]
logger.debug("\tThe last line is %s" %(rawdata_lh[-1]))
logger.debug("\tThe header is %s, and there are %d columns." %(header_lh, len(header_lh)))
# ['id', 'age', 'hhsize', 'numcars', 'password', 'birthyear', 'startyear', 'endyear',
# 'child', 'move', 'edu', 'partner', 'youngchild', 'employ', 'school', 'avail_public',
# 'used_public', 'used_ridehail', 'used_own', 'used_walkbike', 'year', 'decade']
logger.info("\tThere are %d lines in the life history raw dataset." %(len(rawdata_lh)))
id_lh = [item[header_lh.index('id')] for item in rawdata_lh]
logging.info("\tThere are %d unqiue id (person)." %(len(set(id_lh))))

"""Load cross sectional dataset"""
### response for each person per line, 'cs' represents cross sectional.
rawdata_cs = []
with open("/Users/MengqiaoYu/Desktop/WholeTraveler/Data/crossection_full.final.csv", 'r') as input_crosssectional:
    for row in csv.reader(input_crosssectional):
        rawdata_cs.append(row)
header_cs = rawdata_cs[0]
rawdata_cs = rawdata_cs[1:]
logger.debug("The header is %s, and there are %d columns." %(header_cs, len(header_cs)))
logger.info("There are %d lines in the cross sectional dataset." %(len(rawdata_cs)))

id_cs = [item[header_cs.index('id')] for item in rawdata_cs]
assert len(id_cs) == len(set(id_cs)), "Alert: There are duplicate records for one person."

"""Section 1: test the relationship between life events and travel behavior."""
# ### This part only uses life history dataset
# cleandata_lh = remove_na(rawdata_lh)
# behavior_change = find_behavior_change(cleandata_lh)


"""Create a new dataset for model"""
logger.info("(II) Generate a new dataset for people have a full history between age 20-34.")
cleandata_lh = remove_na(rawdata_lh, header_lh) # rawdata_lh is changed = cleandata_lh
cutoff = 1988
cleandata_yound_lh = extract_young_data(cleandata_lh, header_lh, cutoff)
save_model_data(filepath='/Users/MengqiaoYu/Desktop/WholeTraveler/Data/young_data.csv', num_year=2000-cutoff)


"""Merge some info from cross sectional data"""
# Gender?

"""Add policy variables to the dataset"""
# fuel price, unemployment rate
