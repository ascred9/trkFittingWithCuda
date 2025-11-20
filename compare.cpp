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
    const float _ratio = _parMagneticField * 0.299792458;

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
    TBranch* branchPos = trkCuda->FindBranch("pos");
    if (!branchChi2NDF|| !branchR || !branchV)
    {
        std::cout << "Doesn't find the branch" << std::endl;
        return 2;
    }
    float chi2ndf, R, V;
    TVector3* pos = 0;
    trkCuda->SetBranchAddress("chi2ndf", &chi2ndf);
    trkCuda->SetBranchAddress("R", &R);
    trkCuda->SetBranchAddress("V", &V);
    trkCuda->SetBranchAddress("pos", &pos);
    std::cout << "Number of cuda tracks: " << trkCuda->GetEntries() << std::endl;

    TFile* fout = new TFile("compare.root", "RECREATE");
    fout->cd();
    TTree* cmp = new TTree("cmp", "compare original fit and cuda");
    int trkID;
    cmp->Branch("trkID", &trkID);

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

    float dTruth, dOrig, dCuda;
    cmp->Branch("dTruth", &dTruth);
    cmp->Branch("dOrig",  &dOrig);
    cmp->Branch("dCuda",  &dCuda);

    for (int i = 0; i < ntracks; i++)
    {
        trkCuda->GetEntry(i);

        trkID = i;
	    float pt = _ratio * R; // MeV/c
        float pz = _ratio * V; // MeV/c
        if (i < 40)
            std::cout << i << " " << pt << " " << V << " " << pz << std::endl;

        if (_debug)
        {
            std::cout << chi2ndf << " " << pt << " " << pz << " |\t";
            std::cout << tracksFit->at(i)->_chi2ndf << " " << tracksFit->at(i)->_p.Perp() << " " << tracksFit->at(i)->_p.Z() << " |\t";
            std::cout << tracksTruth->at(i)->_chi2ndf << " " << tracksTruth->at(i)->_p.Perp() << " " << tracksTruth->at(i)->_p.Z() << " | ";
            std::cout << std::endl;
        }

        if (tracksFit->at(i)->_p.Mag() < 10)
            continue;

        auto calcCenter = [&](TVector3 stepPos, TVector3 mom)
        {
            TVector3 perp(mom.Y(), -mom.X(), 0);
            perp *= 1.0 / _ratio;
            return stepPos + perp;
        };

        perpTruth = tracksTruth->at(i)->_p.Perp();
        vertTruth = tracksTruth->at(i)->_p.Z();
        dTruth = calcCenter(tracksTruth->at(i)->_pos, tracksTruth->at(i)->_p).Perp() + tracksTruth->at(i)->_p.Perp() / _ratio;

        chi2ndfOrig = tracksFit->at(i)->_chi2ndf;
        perpOrig = tracksFit->at(i)->_p.Perp();
        vertOrig = tracksFit->at(i)->_p.Z();
        dOrig = calcCenter(tracksFit->at(i)->_pos, tracksFit->at(i)->_p).Perp() + tracksFit->at(i)->_p.Perp() / _ratio;

        chi2ndfCuda = chi2ndf;
        perpCuda = pt;
        vertCuda = pz;
        dCuda = pos->Perp() + R;
    
        cmp->Fill();
    }
    fout->cd();
    cmp->Write();
    fout->Close();

    return 0;
}

void draw()
{
    TFile* file = TFile::Open("compare.root");
    TTree* cmp = (TTree*)file->Get("cmp");
    TCanvas* c = new TCanvas("c", "c", 1400, 900);
    c->Divide(2, 1);
    c->cd(1);
    c->cd(1)->SetGrid();
    cmp->Draw("perpCuda - perpTruth>>h1(200, -100, 100)", "", "E");
    cmp->Draw("perpOrig - perpTruth>>h2(200, -100, 100)", "", "E""same");
    gROOT->ProcessLine("h1->SetLineColor(kRed)");
    gROOT->ProcessLine("h1->SetLineWidth(2)");
    gROOT->ProcessLine("h2->SetLineWidth(2)");
    TLegend* legendPerp = new TLegend();
    legendPerp->InsertEntry("h1", "cudafit: Perp difference with truth");
    legendPerp->InsertEntry("h2", "genfit: Perp difference with truth");
    legendPerp->Draw();

    c->cd(2);
    c->cd(2)->SetGrid();
    cmp->Draw("vertOrig - vertTruth>>h4(100, -20, 20)", "", "E");
    cmp->Draw("vertCuda - vertTruth>>h3(100, -20, 20)", "", "E""same");
    gROOT->ProcessLine("h3->SetLineColor(kRed)");
    gROOT->ProcessLine("h3->SetLineWidth(2)");
    gROOT->ProcessLine("h4->SetLineWidth(2)");
    TLegend* legendVert = new TLegend();
    legendVert->InsertEntry("h3", "cudafit: Vert difference with truth");
    legendVert->InsertEntry("h4", "genfit: Vert difference with truth");
    legendVert->Draw();
}

int main(int argc, char** argv)
{
    return process();
}