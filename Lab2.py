import numpy as np
import matplotlib.pyplot as plt

K = int(input('Введите K = ' ))
N = int(input('Введите четное N, которое больше 3 = ' ))

if N < 4 or (N % 2 !=0):
    print('N должно быть четным и больше 3!')
    exit()
   

n = int(N/2)

B = np.random.randint(-10,10,(n,n))
print(f'Подматрица B = \n{B}\n')

C = np.random.randint(0,10,(n,n))
print(f'Подматрица C = \n{C}\n')

D = np.random.randint(-10,10,(n,n))
print(f'Подматрица D = \n{D}\n')

E = np.random.randint(-10,10,(n,n))
print(f'Подматрица E = \n{E}\n')

A = np.vstack([np.hstack([B,C]),np.hstack([D,E])])
print(f'Матрица A = \n{A}\n')

F = A.copy()

EChet = E[:,::2].copy()
ColBolshK=0
for row in range(EChet.shape[0]):
      for x in range(EChet.shape[1]): 
          if EChet[[row], [x]] > K:
                ColBolshK +=1
SumNeChet = np.sum(EChet)


if ColBolshK > SumNeChet:
    print('Кол-во числел больших К в четных стольбцах больше суммы чисел в нечетных столбцах(меняем С и Е симметрично):')
    C1 = np.flip(C, axis=1)
    E1 = np.flip(E, axis=1)
    F = np.vstack([np.hstack([B, E1]), np.hstack([D, C1])])
else:
    print('Сумма числел в нечетных стольбцах больше кол-ва числел больших К в четных стольбцах(меняем В и С несимметрично):')
    F = np.vstack([np.hstack([C, B]), np.hstack([D, E])])
print(F)

if np.linalg.det(A) > (np.diagonal(F).sum() + np.diagonal(np.flip(F, axis=1)).sum()):
    A_trans = np.transpose(A)
    F_trans = np.linalg.transpose(F)
    res = A * A_trans - K * F_trans
else:
    A_trans = np.transpose(A)
    F_inv = np.linalg.inv(F)
    G = np.tril(A)
    G_inv = np.linalg.inv(G)
    res = (A_trans + G_inv - F_inv) * K


print("\nРезультат вычислений:")
print(res)


plt.subplot(2, 2, 1)
plt.imshow(F[:int(n), :int(n)], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 2)
plt.imshow(F[:int(n), int(n):], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 3)
plt.imshow(F[int(n):, :int(n)], cmap='rainbow', interpolation='bilinear')
plt.subplot(2, 2, 4)
plt.imshow(F[int(n):, int(n):], cmap='rainbow', interpolation='bilinear')
plt.show()

plt.subplot(2, 2, 1)
plt.plot(F[:int(n), :int(n)])
plt.subplot(2, 2, 2)
plt.plot(F[:int(n), int(n):])
plt.subplot(2, 2, 3)
plt.plot(F[int(n):, :int(n)])
plt.subplot(2, 2, 4)
plt.plot(F[int(n):, int(n):])
plt.show()