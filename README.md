## Установка

Для работы требуется git, cmake, а также библиотека eigen:
```
apt-get install git cmake libeigen3-dev
```

скачивание и сборка проекта:
```
git clone https://github.com/PgLoLo/chemistry.git
cd chemistry
mkdir build
cd build
cmake ../
make
```

Тепеь можно запустить тесты:
```
./run_tests
```