class Animale:
    def __init__(self,verso,nome,specie,età):
        self.verso = verso
        self.specie = specie
        self.nome = nome
        self.età = età


verso = ['bau','miao','bau','miao']
specie = ['cane','gatto','cane','gatto']
nome = ['fuffi','fido','charly','max']
età = [5,5,5,5]

lista_animali = []
a = Animale
for i in range(4):
    animale_iesimo = a(verso=verso[i],nome=nome[i],specie=specie[i],età=età[i])
    lista_animali.append(animale_iesimo)
