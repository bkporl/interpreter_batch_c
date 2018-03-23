#include<stdio.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<sys/wait.h>
#include<unistd.h>
#define SIZE 30


char* cutcmd(char* ptrInput,int location,int len);
void execute(char* ptrCmd);
int interact();
int batch(int argc,char* argv[]);


int main(int argc,char* argv[])
{

	int sepcmd=0;	
	char* cmd;
	if(argc == 1)
	{
		//interactive
		interact();
		
	}	
	else
	{
		batch(argc,argv);
	}
}	

int interact()
{
		char* cmd;
		char inp[SIZE];

		//int i =0;
		while(1)
		{	
			int sepcmd = 0;
			printf("\n");
			printf(">>");
			gets(inp);// get input
			
			
			if(strcmp(inp,"quit") == 0) return 0;
			else
			{
				
				for(int i=0;i<=strlen(inp);i++)
				{

				  if(i==strlen(inp) ||inp[i] == ';')
				  {
					// pass parameter location inputString and length of command
					cmd = cutcmd(inp,sepcmd+1,i-sepcmd);

					sepcmd = i+1;//sep keep CMDsepCMD, then another CMD
					//once have blank space after ';'
					if(inp[sepcmd]==' ') sepcmd++;		

					execute(cmd);
					
 				  }		

				}
			}
		}
	
	return 0;		
}

int batch(int argc,char* argv[])
{
	FILE* strerr;
	for(int i=1;i<argc;i++)
	{
		char* fileName = argv[i];
		FILE* fp = fopen(fileName,"r");

		if( fp == NULL )
		{ 	
			printf("Error do not open file!");
			exit(1);
		}
		else // opened file
		{	
			
			char str[10];
			int i = 0;

		
			while(1)
			{
				str[i] =  fgetc(fp);
				if (feof(fp)) break;
			
				else
				{
					if ( str[i] == ';' || str[i] == '\n')
					{
 						str[i] = '\0'; // end command ,closed by NULL
						execute(str);
						i=0;
					}
					else i++;
				}
			}
		}
	}
	return 0;
}
void execute(char* ptrCmd)
{
	char* token;
	//get first token point to cmd
	token = strtok(ptrCmd," ");

	char* argv[3];
	char** ptr=argv;
	ptr[0]=NULL; //Program's name
	ptr[1]=NULL;// parameter
	ptr[2]=NULL; // NULL

	int i =0;
	int status;
	while(token != NULL)
	{
		ptr[i] = token;
		i++;
		token = strtok(NULL," ");
	}
	ptr[i] = NULL;
	i=0;

	pid_t pid = fork();
	if(pid == 0)
	{
		execvp(ptr[0],ptr);//exec program
		perror("ERROR");
		exit(1);
	}
	else wait(&status);
}

char* cutcmd(char* ptrInput,int location,int len) //second parameter keep initial address at input
{
	char* ptr;
	ptr= malloc(len+1);
	int i ;
	if(ptr == NULL) 
	{
		printf("allocated fault!");
		exit(1);
	}
	
	
	for (i =0;i<len;i++)
	{	// add address,times by number of command
		*(ptr+i) = *(ptrInput+location-1);
		ptrInput++;
	}
	
	//purpose is to have NULL closed at last address	
	*(ptr+i) = '\0';
	return ptr;
				
			
}
