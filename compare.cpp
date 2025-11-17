#include <iostream>
#include <array>
#include <vector>

#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TGraph2D.h"
#include "TMarker.h"

#include "/home/ascred9/g2esoft/app/include/objects.h"

#include "cuda.h"

int process(bool _debug = false)
{
    const float _parMagneticField = 3.;

    gSystem->Load("/home/ascred9/g2esoft/app/libg2esoftCommon.so");
    TFile* file = TFile::Open("/home/ascred9/g2esoft/testOut.root");
    if (file->IsZombie())
        return 1;

    TTree* trk = (TTree*)file->Get("fitResult");
    TBranch* branchFit = trk->FindBranch("fitTrack");
    TBranch* branchTruth = trk->FindBranch("truthTrack");
    if (!branchFit || !branchTruth)
    {
        std::cout << "Doesn't find the branch" << std::endl;
        return 2;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<g2esoft::Track*>* tracksFit = 0;
    std::vector<g2esoft::Track*>* tracksTruth = 0;
    trk->SetBranchAddress("fitTrack", &tracksFit, TClass::GetClass(typeid(std::vector<g2esoft::Track*>)), kOther_t, true);
    trk->SetBranchAddress("truthTrack", &tracksTruth, TClass::GetClass(typeid(std::vector<g2esoft::Track*>)), kOther_t, true);
    branchFit->GetEntry(0);
    branchTruth->GetEntry(0);
    int ntracks = tracksFit->size();
    std::cout << "Number of init tracks: " << ntracks << " " << tracksTruth->size() << std::endl;

    TFile* fileCuda = TFile::Open("fout.root");
    if (fileCuda->IsZombie())
        return 1;

    TTree* trkCuda = (TTree*)fileCuda->Get("fitCuda");
    TBranch* branchChi2NDF = trkCuda->FindBranch("chi2ndf");
    TBranch* branchR = trkCuda->FindBranch("R");
    TBranch* branchV = trkCuda->FindBranch("V");
    if (!branchChi2NDF|| !branchR || !branchV)
    {
        std::cout << "Doesn't find the branch" << std::endl;
        return 2;
    }
    float chi2ndf, R, V;
    trkCuda->SetBranchAddress("chi2ndf", &chi2ndf);
    trkCuda->SetBranchAddress("R", &R);
    trkCuda->SetBranchAddress("V", &V);
    std::cout << "Number of cuda tracks: " << trkCuda->GetEntries() << std::endl;

    TFile* fout = new TFile("compare.root", "RECREATE");
    fout->cd();
    TTree* cmp = new TTree("cmp", "compare original fit and cuda");
    float chi2ndfOrig, chi2ndfCuda;
    cmp->Branch("chi2ndfOrig", &chi2ndfOrig);
    cmp->Branch("chi2ndfCuda", &chi2ndfCuda);

    float perpTruth, perpOrig, perpCuda;
    cmp->Branch("perpTruth", &perpTruth);
    cmp->Branch("perpOrig", &perpOrig);
    cmp->Branch("perpCuda", &perpCuda);

    float vertTruth, vertOrig, vertCuda;
    cmp->Branch("vertTruth", &vertTruth);
    cmp->Branch("vertOrig", &vertOrig);
    cmp->Branch("vertCuda", &vertCuda);

    for (int i = 0; i < ntracks; i++)
    {
        trkCuda->GetEntry(i);

	    double pt = _parMagneticField * R * 0.299792458; // MeV/c

        if (_debug)
        {
            std::cout << chi2ndf << " " << pt << " " << V << " |\t";
            std::cout << tracksFit->at(i)->_chi2ndf << " " << tracksFit->at(i)->_p.Perp() << " " << tracksFit->at(i)->_p.Z() << " |\t";
            std::cout << tracksTruth->at(i)->_chi2ndf << " " << tracksTruth->at(i)->_p.Perp() << " " << tracksTruth->at(i)->_p.Z() << " | ";
            std::cout << std::endl;
        }

        if (tracksFit->at(i)->_p.Mag() < 10)
            continue;

        perpTruth = tracksTruth->at(i)->_p.Perp();
        vertTruth = tracksTruth->at(i)->_p.Z();

        chi2ndfOrig = tracksFit->at(i)->_chi2ndf;
        perpOrig = tracksFit->at(i)->_p.Perp();
        vertOrig = tracksFit->at(i)->_p.Z();

        chi2ndfCuda = chi2ndf;
        perpCuda = pt;
        vertCuda = V;
    
        cmp->Fill();
    }
    fout->cd();
    cmp->Write();
    fout->Close();

    return 0;
}

int main(int argc, char** argv)
{
    return process();
}