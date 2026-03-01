import os
import numpy as np

#Función para entrada de booleanos
def input_bool(mensaje):
    while True:
        valor = input(mensaje).strip().lower()

        if valor in ("true", "t", "1", "si", "sí", "s", "y", "yes"):
            return True
        elif valor in ("false", "f", "0", "no", "n"):
            return False
        else:
            print("Entrada inválida. Escriba sí/no o true/false.")

# la función limpiar es para limpiar la términal.

def limpiar():
    os.system('cls' if os.name == 'nt' else 'clear')


# Se crea una función llamada valor(), la cual pide como entrada un valor entero, en caso de recibir otro valor no válido, pide al usuario que vuelva a ingresar 
# un valor correcto, por otra parte, si no entra ningún valor, entonces esta asigna un valor establecido.

def valor(mensaje, valor_por_defecto, tipo=float):
    while True:
        entrada = input(f"{mensaje} (default: {valor_por_defecto}): ")
        if entrada == "":
            return valor_por_defecto
        try:
            return tipo(entrada)
        except ValueError:
            print(f"Por favor ingrese un valor válido ({tipo.__name__}).")


def valorint(mensaje, valor_por_defecto, tipo=int):
    while True:
        entrada = input(f"{mensaje} (default: {valor_por_defecto}): ")
        if entrada == "":
            return valor_por_defecto
        try:
            return tipo(entrada)
        except ValueError:
            print(f"Por favor ingrese un valor válido ({tipo.__name__}).")


#════════════════════════════════════════════════════════════════
# Elección del métrica
#════════════════════════════════════════════════════════════════

while True:
    print('Menu de selección de la métrica')
    UserChoice = input('Seleccione una de las siguientes métricas: \n' \
    '1) Métrica de Schwarzschild \n' \
    '2) Métrica de Kerr. \n' \
    '3) Métrica de Minkowski. \n' \
    '4) Métrica de Morris Thorne. \n' \
    '5) Salir. \n' \
    'Opción: '
    )

    if UserChoice == '5':
        break
    elif UserChoice == '1':
        print("Puede escoger el valor de la M.\n"
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "M = 1.")

        M_val = valor('M = ', 1.0)
        metric, r_min = metrica_schwarzschild(M_val)
        limpiar()
        break
        
    elif UserChoice == '2':
        print("Puede escoger el valor de la M y a.\n" #poner nombre a las cosas
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "M = 1 y a = 0.5.")

        M_val = valor('M = ', 1.0)
        a_val = valor('a_val = ', 0.5)
        metric, r_min = metrica_kerr(M_val, a_val)
        limpiar()
        break

    elif UserChoice == '3':
         metric, r_min  = metrica_minkowski()  
         limpiar()
         break

    elif UserChoice == '4':
        print("Puede escoger el valor de b0.\n" #poner nombre a las cosas
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "b0 = 1.")

        b0_val = valor('b0 = ', 1)
        metric, r_min = metrica_morris_thorne(b0_val)
        limpiar()
        break
    
    else:
         print('Solo puede escoger las opciones dadas') 
#════════════════════════════════════════════════════════════════
# Elección del potencial
#════════════════════════════════════════════════════════════════
while True:
    print('Menu de selección del potencial')
    UserChoice = input('Seleccione una de los siguientes potenciales: \n' \
    '1) Potencial Dipolo. \n' \
    '2) Potencial Campo Uniforme. \n' \
    '3) Potencial Coloumb. \n' \
    '4) Potencial Toroidal. \n' \
    '5) Salir. \n' \
    'Opción: '
    )

    if UserChoice == '5':
        break
    elif UserChoice == '1':
        print("Puede escoger el valor de mu .\n"
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "mu = 2.0.")

        mu_val = valor('mu = ', 2.0)
        A_potential = potential_dipolo(mu_val=2.0)
        limpiar()
        break
        
    elif UserChoice == '2':
        print("Puede escoger el valor de E=(a,b,c) y B=(x,y,z).\n" #poner nombre a las cosas
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "E=(a=0, b=0, c=0) y B=(x=0, y=0, z=1).")
        a = valor("a = ", 0)
        b = valor("b = ", 0)
        c = valor("c = ", 0)
        x = valor("x = ", 0)
        y = valor("y = ", 0)
        z = valor("z = ", 1.0)
                  
        A_potential = potential_campo_uniforme(E_val=(a,b,c), B_val=(x,y,z))
        limpiar()
        break

    elif UserChoice == '3':
         print("Puede escoger el valor de Q.\n" #poner nombre a las cosas
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "Q = 1.")
         Q_val = valor('Q = ', 1.0) 
         A_potential = potential_coulomb(Q_val)
         limpiar()
         break

    elif UserChoice == '4':
        print("Puede escoger el valor de B0.\n" #poner nombre a las cosas
      "En caso de no hacerlo se asignarán los valores por defecto:\n"
      "B0 = 1 .")

        B0_val = valor('B0 = ', 1.0)
        A_potential = potential_toroidal(B0_val)
        limpiar()
        break
    
    else:
        print('Solo puede escoger las opciones dadas') 



#════════════════════════════════════════════════════════════════
# Elección parámetros
#════════════════════════════════════════════════════════════════

while True:
    print('═' * 10)
    print('Entrada de parámetros:')
    print('═' * 10)

    CARGADA = input_bool('¿La partícula tiene masa? (sí/no): ')
    if CARGADA == True:
        q_sobre_m = valor('(relación carga/masa) q/m = ', 2.5)
    else:
        q_sobre_m = 0
        
    limpiar()


    print('═' * 10)
    print('\n Sección: Condiciones Iniciales \n')
    print('═' * 10)

    theta0 = valor('Ángulo polar inicial de los rayos: ', np.pi / 2 )
    vtheta0 = valor('Velocidad Inicial: ', 0.0)
    limpiar()

    print('═' * 10)
    print('\n Sección: Integración: \n')
    print('═' * 10)

    N = valorint('N = ', 3000)
    tfinal = valorint('Tiempo Final: ', 80)
    h = tfinal / N 
    r0 = valor('Distancia de Origen de los Rayos: ', 50.0)
    limpiar()

    print('═' * 10)
    print('\n Sección: Rayos: \n')
    print('═' * 10)

    n_rayos = valorint('Número de Rayos ', 60)
    ymin = valorint('Valor mínimo de Rango: ', -10)
    ymax = valorint('Valor máximo de Rango: ', 10)
    y_rango = (ymin, ymax)
    y_minimo = valor('Parámetro de Impacto Mínimo: ', 0.5)
    limpiar()


    print('═' * 10)
    print('\n Sección: Cámara \n')
    print('═' * 10)

    ELEVACION = valorint('Grados Sobre el Plano Ecuatorial: ', 90)
    AZIMUT = valorint('Grados de Rotación Horizontal: ', 270)
    limpiar()

    userchoice = input_bool('¿Desea continuar (1) o reiniciar (0)?')
    if userchoice == True:
        break
    elif userchoice == False:
        pass
    else:
        break

