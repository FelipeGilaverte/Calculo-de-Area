from decimal import Decimal, ROUND_HALF_UP, getcontext

import sympy as sp
from sympy.calculus.util import maximum
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, function_exponentiation, convert_xor


# Aumenta a precisao interna das contas decimais.
getcontext().prec = 50

# Variavel simbolica usada na funcao e nas derivadas.
x = sp.Symbol("x", real=True)

# Permite escrever expressoes como 3x, x^2 e sen(x).
transformacoes = standard_transformations + (implicit_multiplication_application, function_exponentiation, convert_xor)


# log(x) = base 10
# log(x, b) = base b
def log(*args):
    if len(args) == 1:
        return sp.log(args[0], 10)
    if len(args) == 2:
        return sp.log(args[0], args[1])
    raise TypeError("Use log(valor) ou log(valor, base).")


# Nomes matematicos aceitos no input.
nomes = {
    "x": x, "t": x, "e": sp.E, "pi": sp.pi, "abs": sp.Abs, "sen": sp.sin, "sin": sp.sin,
    "cos": sp.cos, "tan": sp.tan, "tg": sp.tan, "sqrt": sp.sqrt, "exp": sp.exp,
    "ln": sp.log, "log": log, "log10": lambda v: sp.log(v, 10),
}


# Le a expressao digitada pelo usuario.
def ler_expressao(mensagem, precisa_ser_numero=False):
    texto = input(mensagem).strip().replace("π", "pi").replace(",", ".").lower()
    expr = sp.simplify(parse_expr(texto, local_dict=nomes, transformations=transformacoes))
    if precisa_ser_numero and getattr(expr, "free_symbols", set()):
        raise ValueError("Valor invalido para o intervalo.")
    return expr

# Converte um resultado do Sympy para Decimal e verifica se ele e real.
def para_decimal(expr):
    valor = sp.N(expr, 25)
    if valor.has(sp.I) or not valor.is_real:
        raise ValueError("A funcao produziu valor nao real no intervalo informado.")
    return Decimal(str(valor))


# Arredondar casas decimais
def arredondar(valor, casas):
    return Decimal(str(valor)).quantize(Decimal("1").scaleb(-casas), rounding=ROUND_HALF_UP)


# Formata os numeros para a saida em portugues.
def formatar(valor, casas=None, tirar_zeros=False):
    valor = Decimal(str(valor))
    texto = format(arredondar(valor, casas), f".{casas}f") if casas is not None else format(valor.normalize(), "f")
    if texto in {"-0", "-0.0"}:
        texto = "0"
    if tirar_zeros and "." in texto:
        texto = texto.rstrip("0").rstrip(".")
    return texto.replace(".", ",")


# Transforma Decimal em racional exato para o Sympy.
def racional(valor):
    n, d = Decimal(str(valor)).as_integer_ratio()
    return sp.Rational(n, d)


# Monta a tabela de pontos x e imagens f(x), ja arredondadas.
def calcular_tabela(funcao, a, h, n, casas):
    xs, fxs = [], []
    for i in range(n + 1):
        xi = a + h * i
        fi = arredondar(para_decimal(funcao.subs(x, racional(xi))), casas)
        xs.append(xi)
        fxs.append(fi)
    return xs, fxs


# Soma as imagens da formula dos trapezios:
# primeira e ultima divididas por 2, as demais inteiras.
def calcular_soma_dos_trapezios(valores_fx):
    soma, parcelas = Decimal("0"), []
    ultimo = len(valores_fx) - 1
    for i, valor in enumerate(valores_fx):
        if i in {0, ultimo}:
            soma += valor / 2
            parcelas.append(f"{formatar(valor, tirar_zeros=True)}/2")
        else:
            soma += valor
            parcelas.append(formatar(valor, tirar_zeros=True))
    return soma, parcelas


# Calcula max|f''(x)| no intervalo.
# Se o Sympy nao conseguir simbolicamente, aproxima numericamente.
def calcular_maximo_modulo_segunda_derivada(f2, a, b):
    if sp.simplify(f2) == 0:
        return Decimal("0")
    try:
        return para_decimal(maximum(sp.Abs(f2), x, sp.Interval(racional(a), racional(b))))
    except Exception:
        g = sp.lambdify(x, sp.Abs(f2), modules=["math"])
        a_float, b_float = float(a), float(b)
        return Decimal(str(max(float(g(a_float + (b_float - a_float) * i / 20000)) for i in range(20001))))


# Calcula o erro de arredondamento usando a formula.
def calcular_erro_de_arredondamento(n, casas, h):
    meia_ultima_casa = Decimal("5").scaleb(-(casas + 1))
    return meia_ultima_casa, Decimal(n) * meia_ultima_casa * h


# Calcula o erro de truncamento usando a segunda derivada.
def calcular_erro_de_truncamento(n, h, max2, casas_trunc):
    erro_bruto = Decimal(n) * (h**3 / Decimal("12")) * max2
    return erro_bruto, arredondar(erro_bruto, casas_trunc)


# Centraliza todos os prints da resolucao.
def mostrar_saida(r):
    print("\nTabela com x e f(x)")
    print(f"{'x':>{r['wx']}} | {'f(x)':>{r['wf']}}")
    print(f"{'-' * r['wx']}-+-{'-' * r['wf']}")
    for vx, vf in zip(r["xt"], r["ft"]):
        print(f"{vx:>{r['wx']}} | {vf:>{r['wf']}}")
    print("\nFuncao interpretada")
    print(f"f(x) = {sp.sstr(r['f'])}")
    print("\nPasso")
    print(f"h = ({r['a']} - {r['b_inicial']})/{r['n']} = {r['h']}")
    print("\nSoma das areas dos trapezios")
    print(f"I aprox. = ({r['h']}) * ({' + '.join(r['parcelas'])})")
    print(f"I aprox. = ({r['h']}) * ({r['soma']}) = {r['area']}")
    print("\nErro de arredondamento")
    print(f"|Ea| <= {r['n']} * {r['meia']} * {r['h']} = {r['ea']}")
    print(f"Modulo do Ea menor ou igual a {r['ea']}")
    print("\nErro de truncamento")
    print(f"f''(x) = {sp.sstr(r['f2'])}")
    print(f"max|f''(x)| em [{r['b_inicial']}; {r['a']}] = {r['max2']}")
    print(f"|ETru| <= {r['n']} * (({r['h']})^3 / 12) * {r['max2']}")
    print(f"|ETru| <= {r['etru_bruto']} aprox. {r['etru']}")
    print("\nErro total")
    print(f"|ETot| <= |Ea| + |ETru| < {r['ea']} + {r['etru']} = {r['etot']}")
    print("\nResposta final")
    print(f"Com parentesis: ({r['area']} +/- {r['etot']})")
    print(f"Com colchetes: [{r['li']}; {r['ls']}]")


def main():
    # Apresentacao inicial.
    print("Programa Integracao Numerica - Metodo dos Trapezios")
    print("Exemplos: 3x + 1, sqrt(x), exp(-x^2/2)/sqrt(2*pi), x^2 sen(1/x^2)")

    # Le a funcao, os extremos do intervalo e os dados do metodo.
    funcao = ler_expressao("\nFuncao f(x): ")
    a = para_decimal(ler_expressao("Extremo a do intervalo [a,b]: ", True))
    b = para_decimal(ler_expressao("Extremo b do intervalo [a,b]: ", True))
    n = int(input("Numero de trapezios: ").strip())
    casas = int(input("Numero de casas decimais: ").strip())
    if b <= a or n <= 0 or casas <= 0:
        raise ValueError("Entradas invalidas.")

    # Calcula o passo h e monta a tabela.
    h = (b - a) / Decimal(n)
    xs, fxs = calcular_tabela(funcao, a, h, n, casas)
    soma, parcelas = calcular_soma_dos_trapezios(fxs)
    area = arredondar(h * soma, casas)

    # Calcula a segunda derivada e o maximo do seu modulo.
    f2 = sp.simplify(sp.diff(funcao, x, 2))
    max2 = calcular_maximo_modulo_segunda_derivada(f2, a, b)
    # Calcula Ea, ETru, ETot e o intervalo final da resposta.
    meia, ea = calcular_erro_de_arredondamento(n, casas, h)
    casas_trunc = casas + 1
    etru_bruto, etru = calcular_erro_de_truncamento(n, h, max2, casas_trunc)
    casas_resposta = max(casas, casas_trunc)
    etot = arredondar(ea + etru, casas_resposta)
    li = arredondar(area - etot, casas_resposta)
    ls = arredondar(area + etot, casas_resposta)

    # Prepara os textos ja formatados para a impressao final.
    xt = [formatar(v, tirar_zeros=True) for v in xs]
    ft = [formatar(v, casas) for v in fxs]
    mostrar_saida({
        "f": funcao, "f2": f2, "a": formatar(b, tirar_zeros=True), "b_inicial": formatar(a, tirar_zeros=True), "n": n,
        "h": formatar(h, tirar_zeros=True), "parcelas": parcelas, "soma": formatar(soma, casas), "area": formatar(area, casas),
        "meia": formatar(meia, tirar_zeros=True), "ea": formatar(ea, tirar_zeros=True), "max2": formatar(max2, tirar_zeros=True),
        "etru_bruto": formatar(etru_bruto, casas + 1), "etru": formatar(etru, casas_trunc), "etot": formatar(etot, casas_resposta),
        "li": formatar(li, casas_resposta, True), "ls": formatar(ls, casas_resposta, True), "xt": xt, "ft": ft,
        "wx": max(len("x"), *(len(v) for v in xt)), "wf": max(len("f(x)"), *(len(v) for v in ft)),
    })


if __name__ == "__main__":
    # Executa o programa e mostra uma mensagem simples em caso de erro.
    try:
        main()
    except Exception as erro:
        print(f"\nErro: {erro}")
