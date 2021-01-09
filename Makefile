activate_venv:
	(source venv/bin/activate)
	pip install -r  requirements.txt

install: test
	pip install ./

build: test
	python setup.py bdist_wheel

test: activate_venv
	coverage run --omit 'venv/*' -m pytest test
	coverage-badge -fo coverage.svg

clean:
	rm -rf build
	rm -rf dist
	rm -rf meta_act.egg-info
