
STD = c++17

all: quetza

quetza: build/main.o build/lexer.o | build
	$(CXX) build/main.o build/lexer.o -O3 -o quetza

build:
	@ if [ ! -d build ]; then mkdir build; fi

build/main.o: main.cpp | build
	$(CXX) -std=$(STD) -c -O3 main.cpp -fno-exceptions -fPIC -o build/main.o -I .

build/lexer.o: lexer.h lexer.cpp | build
	$(CXX) -std=$(STD) -c -O3 lexer.cpp -fno-exceptions -fPIC -o build/lexer.o -I .

.PHONY = clean
clean: build
	@ rm build/*.o