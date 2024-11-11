# CFD_Riemann_solver
Riemann solver for Euler equation implementation with chemistry (Computational Fluid Dynamics course, MIPT 5)

Всё взято из Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics" (third edition), главы 4 и 10
В комментариях постаралась оставить ссылки на все используемые уравнения 
Изначально код списан с https://github.com/pmocz/riemann-solver/tree/main

riemann_solver_with_animation.py:
    Выводит гифку Animation.gif для плотности, скорости, давления и температуры.
    Базово - для н.у. из теста Сода
    Строки с н.у. для химической реакции закомментированы - для таких значений нужно менять шаг по пространству и число Куранта, и пока для таких чисел строится невправильно

my_riemann_solver.py:
    то же самое, но без температуры и только с обычными н.у.
    Выводит интерактивные графики. Точное решение считается быстро, для численного решения - график обновляется долго, но обновляется
    Какое решение выводится (точное или численное) - определяется в последних двух строчках



