from basix import ElementFamily, LagrangeVariant


def family_by_name(family):
    families = {
        "Lagrange": ElementFamily.P,
        "BDM": ElementFamily.BDM,
        "RT": ElementFamily.RT,
        "N1E": ElementFamily.N1E,
        "N2E": ElementFamily.N1E,
    }
    return families[family]


def basis_variant():
    return LagrangeVariant.gll_centroid
