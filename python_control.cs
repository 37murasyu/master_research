using UnityEngine;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

public class RunPythonScript : MonoBehaviour
{
    public string pythonInterpreter = "C:/Users/villa/AppData/Local/Programs/Python/Python39/python.exe";
    public string pythonScriptPath = "C:/Users/villa/My project (2)/Assets/GoTounity/master_research_code.py";

    private bool isActionActive = false;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            isActionActive = !isActionActive;
            if (isActionActive)
            {
                UnityEngine.Debug.Log("python start");

                RunPythonAsync();
            }
        }
    }

    async void RunPythonAsync()
    {
        ProcessStartInfo start = new ProcessStartInfo()
        {
            FileName = pythonInterpreter,
            Arguments = $"\"{pythonScriptPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };

        using (Process process = new Process { StartInfo = start })
        {
            process.Start();
            string output = await process.StandardOutput.ReadToEndAsync();
            process.WaitForExit();  // èIóπÇë“Ç¬
            UnityEngine.Debug.Log("Python script output: " + output);
            UnityEngine.Debug.Log("Python script finished with exit code " + process.ExitCode);
        }
    }
}
