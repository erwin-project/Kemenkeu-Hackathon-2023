import re
import warnings
import json
import numpy as np
import ogr
import geopandas as gpd
import subprocess
warnings.filterwarnings("ignore")


def update_json(name, username, email, password):
    data = open('data/data_account.json')

    data_account = json.load(data)

    name = data_account['name'] + [name]
    username = data_account['username'] + [username]
    email = data_account['email'] + [email]
    password = data_account['password'] + [password]

    data.close()

    data_email = {'name': name,
                  'username': username,
                  'email': email,
                  'password': password}

    with open('data/data_account.json', 'w') as json_file:
        json.dump(data_email, json_file)

    return None


def replace_json(name, username, old_email, new_email, password):
    data = open('data/data_account.json')

    data_account = json.load(data)

    index = np.where(np.array(data_account['email']) == old_email)[0][0]
    data_account['name'][index] = name
    data_account['username'][index] = username
    data_account['email'][index] = new_email
    data_account['password'][index] = password

    data.close()

    data_email = {'name': data_account['name'],
                  'username': data_account['username'],
                  'email': data_account['email'],
                  'password': data_account['password']}

    with open('data/data_account.json', 'w') as json_file:
        json.dump(data_email, json_file)

    return None


def check_account(name_email, name_password):
    data = open('data/data_account.json')

    data_email = json.load(data)

    name = data_email['name']
    username = data_email['username']
    email = data_email['email']
    password = data_email['password']

    index = np.where(np.array(email) == name_email)[0][0]
    password_true = password[index]

    if name_email in email and name_password == password_true:
        return name[index], username[index], 'register'
    if name_email in email and name_password != password_true:
        return '', '', 'wrong password'
    if name_email not in email:
        return '', '', 'not register'


def check_email(email):
    data = open('data/data_account.json')

    data_email = json.load(data)

    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    if re.fullmatch(regex, email):
        if email not in data_email['email']:
            value = "valid email"
        else:
            value = "duplicate email"
    else:
        value = "invalid email"

    return value


def change_province(data):
    data.rename(columns={"PROVINSI": "Provinsi"},
                inplace=True)

    data['Provinsi'] = data['Provinsi'].str.title()

    data['Provinsi'] = data['Provinsi'].replace(['Daerah Istimewa Yogyakarta', 'Dki Jakarta',
                                                 'Kepulauan Bangka Belitung', 'Kepulauan Riau'],
                                                ['DI Yogyakarta', 'DKI Jakarta',
                                                 'Kep. Bangka Belitung', 'Kep. Riau'])

    return data


def change_json(path):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_path = path
    data_source = driver.Open(shp_path, 0)

    fc = {
        'type': 'FeatureCollection',
        'features': []
    }

    lyr = data_source.GetLayer(0)
    for feature in lyr:
        fc['features'].append(feature.ExportToJson(as_object=True))

    with open('data/Indonesia_SHP.json', 'w') as f:
        json.dump(fc, f)

    data_json = gpd.read_file('data/Indonesia_SHP.json')

    return data_json


def unit_data(value):
    unit = {'Pendapatan_per_Kapita_per_bulan': 'Rupiah',
            'Anggaran_Kesehatan_per_Kapita': 'Rasio Persentase',
            'Anggaran_Kesehatan': 'Rasio Persentase',
            'Cakupan_JKN': 'Rasio Persentase',
            'TTRS_per_1000': 'Satuan per 1000',
            'Tenaga_Medis_per_10k_Populasi': 'Satuan per 10000',
            'AHH': 'Usia',
            'Total_Efisiensi': 'Rasio Persentase',
            'Pertumbuhan_Cakupan_JKN': 'Rasio Persentase',
            'Pertumbuhan_TTRS_per_1000': 'Rasio Persentase',
            'Pertumbuhan_Tenaga_Medis_per_10k': 'Rasio Persentase',
            'Pertumbuhan_AHH': 'Rasio Persentase',
            'Efisiensi_Cakupan_JKN': 'Rasio Persentase',
            'Efisiensi_TTRS_per_1000': 'Rasio Persentase',
            'Efisiensi_Tenaga_Medis_per_10k': 'Rasio Persentase',
            'Efisiensi_AHH': 'Rasio Persentase'}

    return unit[value]

