# 1) Define your expected regex per column (anchor them!)
"""
transactions_patterns = {
    "Point_de_Vente":         r"^PDV-id-\d{4,5}$",          # "PDV-id-0063", "PDV-id-00592"
    "Numero_TPV":             r"^TPV_\d{3}$",               # "TPV_155", "TPV_925"
    "Numero_Transaction":     r"^TID\d+$",                  # "TID000006400021"
    "Date_Transaction":       r"^\d{4}-\d{2}-\d{2}$",       # 2022-03-30
    "Heure":                  r"^\d{2}:\d{2}:\d{2}$",       # 12:00:00
    "Typologie_Magasin":      r"^Typologie_Magasin_\d+$",   # "Typologie_Magasin_05"
    "Numero_Fidelite":        r"^N_\d+$",                   # "N_3067989"
    "Type_de_Vente":          r"^TV\d+$",                   # "TV3"
    "Univers_Produit":        r"^CL1_\d+$",                 # "CL1_10"
    "Segment_Produit":        r"^CL2_\d+$",                 # "CL2_109"
    "Famille_Produit":        r"^CL3_\d+$",                 # "CL3_374"
    "Sous_Famille_Produit":   r"^CL4_\d+$",                 # "CL4_453"
    "Fedas_Numero":           r"^FedasNum\d+$",             # "FedasNum434"
    "Fedas_Libelle":          r"^FedasLib\d+$",             # "FedasLib2124"
    "Cible_Genre_Age":        r"^CGA\d+$",                  # "CGA6"
    "Modele_Couleur_Ref":     r"^MCR\d+$",                  # "MCR203824"
    "Modele_Couleur_Libelle": r"^MCL\d+$",                  # "MCL27918"
    "Type_de_vente_NPS":      r"^NPS\d+$",                  # "NPS2"
    "Quantite_Vendue":        r"^\d+$",                     # Integer-typed quantity in your sample.
    "CA_Net_HT":              r"^\d+(?:.\d+)?$",            # String-typed monetary column in your sample (European decimal comma).
    "CA_Net_TTC":             r"^\d+(?:.\d+)?$",            # String-typed monetary column in your sample (European decimal comma).
    "Marge_Nette_Magasin":    r"^\d+(?:.\d+)?$",            # String-typed monetary column in your sample (European decimal comma).
}"""


transactions_patterns = {
    "Point_de_Vente":         r"^PDV-id-\d+$",          # "PDV-id-0063", "PDV-id-00592"
    "Numero_TPV":             r"^TPV_\d+$",               # "TPV_155", "TPV_925"
    "Numero_Transaction":     r"^TID\d+$",                  # "TID000006400021"
    "Date_Transaction":       r"^\d{4}-\d{2}-\d{2}$",       # 2022-03-30
    "Heure":                  r"^\d{2}:\d{2}:\d{2}$",       # 12:00:00
    "Typologie_Magasin":      r"^Typologie_Magasin_\d+$",   # "Typologie_Magasin_05"
    "Numero_Fidelite":        r"^N_\d+$",                   # "N_3067989"
    "Type_de_Vente":          r"^TV\d+$",                   # "TV3"
    "Univers_Produit":        r"^CL1_\d+$",                 # "CL1_10"
    "Segment_Produit":        r"^CL2_\d+$",                 # "CL2_109"
    "Famille_Produit":        r"^CL3_\d+$",                 # "CL3_374"
    "Sous_Famille_Produit":   r"^CL4_\d+$",                 # "CL4_453"
    "Fedas_Numero":           r"^FedasNum\d+$",             # "FedasNum434"
    "Fedas_Libelle":          r"^FedasLib\d+$",             # "FedasLib2124"
    "Cible_Genre_Age":        r"^CGA\d+$",                  # "CGA6"
    "Modele_Couleur_Ref":     r"^MCR\d+$",                  # "MCR203824"
    "Modele_Couleur_Libelle": r"^MCL\d+$",                  # "MCL27918"
    "Type_de_vente_NPS":      r"^NPS\d+$",                  # "NPS2"
    "Quantite_Vendue":        r"^[+-]?\d+$",                # Integer-typed quantity in your sample.
    "CA_Net_HT":              r"^[+-]?\d+(?:[.,]\d+)?$",          # String-typed monetary column in your sample (European decimal comma).
    "CA_Net_TTC":             r"^[+-]?\d+(?:[.,]\d+)?$",           # String-typed monetary column in your sample (European decimal comma).
    "Marge_Nette_Magasin":    r"^[+-]?\d+(?:[.,]\d+)?$",           # String-typed monetary column in your sample (European decimal comma).
}

products_patterns = {
    "CODE_MODELE_COULEUR_ACTUEL": r"^CMC\d+$",
    "SIGNATURE_PRODUCT":          r"^(?:0|1|)$",  # allow "", "0", "1"
}



