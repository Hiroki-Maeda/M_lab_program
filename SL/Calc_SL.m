%パスを通す。Calc_SL.m，lab_SL.m
%type : どの周波数体なのかを指定。a1,a2,b,y,dl,thの文字列
function[] = Calc_SL(type,ch)
  if(exist('ch') == 0) ch =19; end %ch数の指定がなければ19ch
  dirMove = 0;
  dirExists = 0;
  files = dir();
  for i=1:length(files)
      if(strcmp(files(i).name ,type))
          dirExists = 1;
      end
  end
  if(exist(type) == 7 &&  dirExists == 1)
      cd(type);
      dirMove = 1;
  else
      disp('No Directry!!')
      return
  end
  %#### 定数指定 ######
    pref = 0.01;
    speed = 16;
  %####################

  files = dir(strcat('*',type,'.txt*'));
  filesNum = length(files);
  if(filesNum == 0)
    disp('  check type!!!!!');
  end
  fileID = fopen('output.csv','w');

  for i=1:filesNum
    disp(strcat('----------------time::',num2str(i),'----------------------------'));
    s = load(files(i).name);
    [r c] = size(s);
    if(c ~= ch) 
      disp('check col length!!')
      fclose(fileID);
      break;
    end
    [lag m w1 w2 check] = getParameta4CalcSL(type);
    if(check ~= -1)
      disp('');disp('  #####Calc_SL######');
      [result_SL1 result_SL2]=lab_SL2(s,lag,m,w1,w2,pref,speed);
      disp('  ##################');disp('');
    else
        fclose(fileID);
        disp('No Type!!!!')
        return
    end
    saveFileName = strrep(files(i).name,'.txt','_SL.csv')
    csvwrite(saveFileName,result_SL1); 
    [~,~,z] = size(result_SL1);
    files(i).z= z;
    
    fprintf(fileID,'%s%s %d\n',saveFileName,',',files(i).z);
    disp('---------------------------------------------------');
  end
  fclose(fileID);
  if(dirMove == 1)
     cd('..');
  end
end

function[lag m w1 w2 check] = getParameta4CalcSL(type)
  check = 1;

  if(type == 'a1')
    lag=33; m=5;w1=264; w2=1263;
  elseif(type == 'a2')
    lag=26; m=5;w1=208; w2=1207;
  elseif(type == 'b')
    lag=11; m=8;w1=154; w2=1153;
  elseif(type == 'y')
    lag=7; m=6;w1=70; w2=1069;
  elseif(type == 'dl')
    lag=83; m=25;w1=3984; w2=4983; %20190703までw2=4183だった
  elseif(type == 'th')
    lag=42; m=7;w1=504; w2=1503;
  else
    check = -1;
    disp('  check type!!!!!')
  end
end