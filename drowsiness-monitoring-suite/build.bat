@echo off
echo ===================================
echo �����茟�m�V�X�e�� �r���h�X�N���v�g
echo ===================================

echo �K�v�ȃ��C�u�������C���X�g�[�����Ă��܂�...
pip install -r requirements.txt
pip install pyinstaller

echo EXE�t�@�C�����r���h���Ă��܂�...
pyinstaller drowsiness_watcher.spec

echo �r���h�����I
echo dist\drowsiness_watcher.exe ���쐬����܂���

echo ���������t�@�C�����f�X�N�g�b�v�ɃR�s�[���܂����H (Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    echo �f�X�N�g�b�v�ɃR�s�[���Ă��܂�...
    copy dist\drowsiness_watcher.exe %USERPROFILE%\Desktop\
    copy .env.template %USERPROFILE%\Desktop\.env.template
    echo �R�s�[�����I
    echo �f�X�N�g�b�v�� .env.template �� .env �Ƀ��l�[�����Đݒ��ҏW���Ă��������B
)

echo �r���h�v���Z�X���������܂����B
pause