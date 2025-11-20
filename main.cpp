#include <iostream>
#include <array>
#include <vector>
#include <deque>

#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph2D.h"
#include "TMarker.h"

#include "/home/ascred9/g2esoft/app/include/objects.h"

#include "cuda.h"

int process()
{
    gSystem->Load("/home/ascred9/g2esoft/app/libg2esoftCommon.so");
    //TFile* file = TFile::Open("testOut.root");
    TFile* file = TFile::Open("/home/ascred9/g2esoft/testOut.root");
    if (file->IsZombie())
        return 1;

    TTree* trk = (TTree*)file->Get("trk");
    TBranch* branch = trk->FindBranch("FoundTracks");
    if (!branch)
    {
        std::cout << "Doesn't find the branch" << std::endl;
        return 2;
    }

    std::cout << "Fill data" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<const g2esoft::Track*>* tracks = 0;
    trk->SetBranchAddress("FoundTracks", &tracks, TClass::GetClass(typeid(std::vector<const g2esoft::Track*>)), kOther_t, true);
    branch->GetEntry(0);
    int ntracks = tracks->size();
    const std::vector<const g2esoft::RecoHit*>* recoHits;

    auto isGhosts = [](const std::vector<const g2esoft::RecoHit*>* hits, int ihit, int jhit)
    {
        if (abs(hits->at(ihit)->_pos.Phi() - hits->at(jhit)->_pos.Phi()) > 0.05)
            return false;

        if ((hits->at(ihit)->_pos - hits->at(jhit)->_pos).Mag() > 5)
            return false;

        return true;
    };

    std::vector<std::pair<int, std::vector<int>>> tracksToFit;
    for (int itrk = 0; itrk < ntracks; itrk++)
    {
        recoHits = &tracks->at(itrk)->_recoHits;
        int nhits = recoHits->size();
        if (nhits < 4)
            continue;

        std::vector<int> perfectVec = {0};
        for (int ihit = 0; ihit < nhits; ihit++)
        {
            bool toAdd = true;
            for (auto jit = perfectVec.begin(); jit != perfectVec.end(); ++jit)
                toAdd &= !isGhosts(recoHits, ihit, *jit);

            if (toAdd)
                perfectVec.push_back(ihit);

            if (perfectVec.size() == 4)
                break;
        }

        if (perfectVec.size() != 4) // add all
            perfectVec = {nhits - 4, nhits - 3, nhits - 2, nhits - 1};
    
        std::vector<std::pair<int, std::vector<int>>> tmp;
        for (int k = 0; k <= perfectVec.at(0); k++)
            for (int l = k + 1; l <= perfectVec.at(1); l++)
                for (int m = l + 1; m <= perfectVec.at(2); m++)
                    for (int n = m + 1; n <= perfectVec.at(3); n++)
                        tmp.push_back( {itrk, {k, l, m, n}} );
    
        /*
        if (tmp.size() > 1)
            for (auto it = tmp.begin(); it != tmp.end(); ++it)
                std::cout << (*it).first << ": " << (*it).second.at(0) << " " << (*it).second.at(1) << " " << (*it).second.at(2)<< " " << (*it).second.at(3) << std::endl;
        */

        tracksToFit.insert(tracksToFit.end(), tmp.begin(), tmp.end());
    }
    
    std::cout << "Old size: " << ntracks << " and new size: " << tracksToFit.size() << std::endl;
    ntracks = tracksToFit.size();

    // Sort all vectors
    /*
    for (int i = 0; i < ntracks; i++)
    {
        recoHits = &tracks->at(tracksToFit.at(i).first)->_recoHits;
        std::sort(tracksToFit.at(i).second.begin(), tracksToFit.at(i).second.end(),
        [&recoHits](int l, int r){
            return abs(recoHits->at(l)->_pos[2] < abs(recoHits->at(r)->_pos[2]));
        });
    }
    */

    float* data = new float[ntracks * 4 * 3];
    int size = 0;
    for (int i = 0; i < ntracks; i++)
    {
        recoHits = &tracks->at(tracksToFit.at(i).first)->_recoHits;
        int nhits = recoHits->size();
        if (nhits < 4)
            continue;

        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                data[i * 12 + j * 3 + k] = recoHits->at(tracksToFit.at(i).second.at(j))->_pos[k];
        
        size++;
    }

    std::cout << "Size: " << size << " -> " << size*12 << std::endl;
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "time: " << duration.count()/1e6 << " sec" << std::endl;
    std::cout << "Finish filling" << std::endl;

    std::cout << "Launch cuda" << std::endl;
    printDeviceInfo(0);
    start = std::chrono::high_resolution_clock::now();
    int npars = 5, ninfo = 3;
    float* output = new float[size * npars]; // (x0, y0, z0), R, Vz
    float* info = new float[size * ninfo]; // (chiSquare, last chiSquare gradient, initial gradient)

    fitTracks(data, size, output, info);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "time: " << duration.count()/1e6 << " sec" << std::endl;
    std::cout << "Finish cuda" << std::endl;

    // Save data
    TFile* fout = new TFile("fout.root", "RECREATE");
    TTree* fitCuda = new TTree("fitCuda", "fit by CUDA");
    float chi2ndf, R, V;
    TVector3 pos;
    fitCuda->Branch("chi2ndf", &chi2ndf);
    fitCuda->Branch("R", &R);
    fitCuda->Branch("V", &V);
    fitCuda->Branch("pos", &pos);
    for (int trkID = 0; trkID < ntracks; trkID++)
    {
        if (trkID < ntracks - 1 && tracksToFit.at(trkID).first == tracksToFit.at(trkID + 1).first) //duplicate
        {
            // choose best chi2
            int bestID = trkID;
            float bestChi = info[ninfo * trkID];
            while (trkID < ntracks - 1 && tracksToFit.at(trkID).first == tracksToFit.at(trkID + 1).first)
            {
                ++trkID;
                if (std::isnan(info[ninfo * trkID]) || info[ninfo * trkID] < 0)
                    continue;

                if (bestChi > info[ninfo * trkID])
                {
                    bestID = trkID;
                    bestChi = info[ninfo * trkID];
                }
            }

            chi2ndf = info[ninfo * bestID];
            R = output[npars * bestID + 3];
            V = output[npars * bestID + 4];
            pos = TVector3(output[npars * bestID], output[npars * bestID + 1], output[npars * bestID + 2]);
            fitCuda->Fill();
            continue;
        }

        chi2ndf = info[ninfo * trkID];
        R = output[npars * trkID + 3];
        V = output[npars * trkID + 4];
        pos = TVector3(output[npars * trkID], output[npars * trkID + 1], output[npars * trkID + 2]);
        fitCuda->Fill();
    }
    fitCuda->Write();

    // Vizualization
    for (int trkID = 300; trkID < 400; trkID++)
    {
        std::cout << trkID << " (";
        std::cout << tracksToFit.at(trkID).first << ") ";
        for (int j = 0; j < 5; j++)
            std::cout << output[npars * trkID + j] << " ";

        std::cout << ", chiSquare: " << info[ninfo * trkID] << ", abs sum Jacobian: " << info[ninfo * trkID + 1] << ", init chiSquare: " << info[ninfo * trkID + 2];
        std::cout << std::endl;

        TCanvas c(Form("c%d_%d", trkID, tracksToFit.at(trkID).first), Form("c%d", trkID), 900, 900);

        float x0 = output[npars * trkID];
        float y0 = output[npars * trkID + 1];
        float z0 = output[npars * trkID + 2];
        float R = output[npars * trkID + 3];
        float V = output[npars * trkID + 4];

        TGraph2D graphBox(2);
        graphBox.SetName("box");
        graphBox.SetPoint(0, -400, -400, -200);
        graphBox.SetPoint(1, 400, 400, 200);
        graphBox.Draw("P");

        TGraph2D graphB(100);
        graphB.SetName("beam trajectory");
        for (int j = 0; j <= 100; j++)
        {
            double t = -TMath::Pi() + j/100. * 2 * TMath::Pi();
            double x = 333 * cos(-t);
            double y = 333 * sin(-t);
            double z = 0;
            graphB.SetPoint(j, x, y, z);
        }
        graphB.SetLineColor(kBlue);
        graphB.SetMinimum(-200.);
        graphB.SetMaximum(200.);
        graphB.Draw("LINE""SAME");

        TGraph2D graph2(100);
        graph2.SetName("trajectory");
        for (int j = 0; j <= 100; j++)
        {
            double t = -2 * TMath::Pi() + j/100. * 6 * TMath::Pi();
            double x = x0 + R * cos(-t);
            double y = y0 + R * sin(-t);
            double z = z0 + V * t;
            graph2.SetPoint(j, x, y, z);
        }

        if (R > 0)
            graph2.Draw("LINE""SAME");

        TGraph2D graph(4);
        graph.SetName("data");
        if (R > 0)
            graph.Draw("P""SAME");
        else
            graph.Draw("P");
        graph.SetName(Form("gr%d", trkID));
        graph.SetMinimum(-200.);
        graph.SetMaximum(200.);
        graph.GetXaxis()->SetTitle("X, mm");
        graph.GetYaxis()->SetTitle("Y, mm");
        graph.GetZaxis()->SetTitle("Z, mm");
        graph.SetMarkerStyle(20);

        for (int j = 0; j < 4; j++)
        {
            double x = data[trkID * 12 + j * 3 + 0];
            double y = data[trkID * 12 + j * 3 + 1];
            double z = data[trkID * 12 + j * 3 + 2];
            graph.SetPoint(j, x, y, z);
        }
        
        double x = data[trkID * 12 + 0];
        double y = data[trkID * 12 + 1];
        double z = data[trkID * 12 + 2];
        TGraph2D graph3(1);
        graph3.SetName("point");
        graph3.SetMarkerStyle(20);
        graph3.SetMarkerColor(kRed);
        graph3.SetPoint(0, x, y, z);
        graph3.Draw("P""SAME");

        gPad->Modified(); gPad->Update();
        c.Write();
    }
    fout->Close();
    
    delete[] data;
    delete[] output;

    return 0;
}

int main(int argc, char** argv)
{
    return process();
}