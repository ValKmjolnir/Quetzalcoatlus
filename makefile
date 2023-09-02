
STD = c++17

all: quetza

quetza: build/main.o build/lexer.o | build
	$(CXX) build/main.o build/lexer.o -O3 -o quetza

build:
	@ if [ ! -d build ]; then mkdir build; fi

build/main.o: src/main.cpp | build
	$(CXX) -std=$(STD) -c -O3 src/main.cpp -fno-exceptions -fPIC -o build/main.o -I .

build/lexer.o: src/lexer.h src/lexer.cpp | build
	$(CXX) -std=$(STD) -c -O3 src/lexer.cpp -fno-exceptions -fPIC -o build/lexer.o -I .

.PHONY = clean
clean: build
	@ rm build/*.o