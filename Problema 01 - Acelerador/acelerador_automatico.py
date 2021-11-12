import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


# ENTRADAS 
vel= ctrl.Antecedent(np.arange(0, 120), 'velocidade')
dist = ctrl.Antecedent(np.arange(0, 20), 'distancia_carro_frente')

# CONSEQUENCIAS
cmd = ctrl.Consequent(np.arange(0, 100), 'comando')

# FUNCOES DE VELOCIDADE
vel['lento'] = fuzz.trapmf(vel.universe, [0, 0, 30, 60])
vel['medio'] = fuzz.trapmf(vel.universe, [40, 50, 70, 80])
vel['rapido'] = fuzz.trapmf(vel.universe, [70 , 90, 120,120])

# FUNCOES DE DISTANCIA
dist['longe'] = fuzz.trapmf(dist.universe, [5,15,20,20])
dist['proximo'] = fuzz.trapmf(dist.universe, [0,0,3,10])

# FUNCOES DE COMANDOS
cmd['acelerar'] = fuzz.trapmf(cmd.universe, [0, 0, 50, 60])
cmd['freiar'] = fuzz.trapmf(cmd.universe, [60,90,100,100])

# REGRAS
rule1 = ctrl.Rule(
	(vel['lento'] & dist['longe']) |
	(vel['lento'] & dist['proximo']) |
	(vel['medio'] & dist['longe']), 
	cmd['acelerar']
)
rule2 = ctrl.Rule(
	(vel['rapido'] & dist['proximo']),
	cmd['freiar']
)

# CRIANDO O SISTEMA E SIMULANDO
cmd_ctrl = ctrl.ControlSystem([rule1, rule2])
cmd_output = ctrl.ControlSystemSimulation(cmd_ctrl)

# IMPRIMINDO
'''
cmd.view(sim=cmd_output)
plt.show()

vel.view()
plt.show()

dist.view()
plt.show()
'''

# INPUTS
velocidade = float(input("Velocidade (0 a 120 km/h): "))
while velocidade < 0 or velocidade > 120:
	try:
		velocidade = float(input("Velocidade (0 a 120 km/h): "))
	except ValueError:
		print('Valor inválido')

distancia = float(input("Distância do carro da frente (0 a 20 metros): "))
while distancia < 0 or distancia > 20:
	try:
		distancia = float(input("Distância do carro da frente (0 a 20 metros): "))
	except ValueError:
		print('Valor inválido')

# ADICIONANDO AO CONTROLADOR
cmd_output.input['velocidade'] = velocidade
cmd_output.input['distancia_carro_frente'] = distancia

# COMPUTANDO
cmd_output.compute()

# APRESENTANDO RESULTADO
cmd.view(sim=cmd_output)
plt.show()
print(cmd_output.output['comando'])
#'''