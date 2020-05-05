DIR := ${CURDIR}
all:
	@echo "See Makefile for possible targets!"

purepospy/purepos-2.1.one-jar.jar:
	wget https://github.com/ppke-nlpg/purepos/releases/download/v2.1/purepos-2.1.one-jar.jar \
        -O purepospy/purepos-2.1.one-jar.jar
	mkdir purepospy/purepos-2.1.one-jar && cd purepospy/purepos-2.1.one-jar && jar xvf ../purepos-2.1.one-jar.jar && \
        cd ../..

download-purepos: purepospy/purepos-2.1.one-jar.jar

dist/*.whl dist/*.tar.gz: download-purepos
	@echo "Building package..."
	python3 setup.py sdist bdist_wheel

build: dist/*.whl dist/*.tar.gz

install-user: build
	@echo "Installing package to user..."
	pip3 install dist/*.whl

test:
	@echo "Running tests..."
	cd /tmp && python3 -m purepospy -i $(DIR)/tests/postag_kutya.in | diff - $(DIR)/tests/postag_kutya.out && \
        cd ${CURDIR}

install-user-test: install-user test
	@echo "The test was completed successfully!"

ci-test: install-user-test

uninstall:
	@echo "Uninstalling..."
	pip3 uninstall -y purepospy

install-user-test-uninstall: install-user-test uninstall

clean:
	rm -rf dist/ build/ purepospy.egg-info/
	rm -rf purepospy/purepos-2.1.one-jar.jar purepospy/purepos-2.1.one-jar

clean-build: clean build
